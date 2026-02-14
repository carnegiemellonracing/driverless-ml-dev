#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <cstdlib>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorData.hpp>

using namespace nvinfer1;

struct Detection
{
    cv::Rect_<float> rect;
    float prob;
    int label;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kVERBOSE) {
            std::cout << msg << "\n";
        }
    }
} gLogger;

class YOLODetector {
public:
    YOLODetector(std::string engine_file_path);
    ~YOLODetector();
    std::vector<Detection> detect (const cv::Mat& img, float conf);

private:
    void preprocessCudaToInputMem(const cv::Mat& img);

    Logger logger;
    ICudaEngine* engine;
    IRuntime* runtime;
    IExecutionContext* context;
    cudaStream_t stream = nullptr;

    std::unique_ptr<cvcuda::ResizeCropConvertReformat> preprocess_op;

    nvcv::Tensor input_tensor;
    nvcv::Tensor output_tensor;

    void* d_input_u8 = nullptr;
    void* input_mem = nullptr;
    void* output_mem = nullptr;

    int last_input_width = 0;
    int last_input_height = 0;
    static constexpr int MODEL_W = 640;
    static constexpr int MODEL_H = 640;

    static const int INPUT_SIZE = 1 * 3 * 640 * 640 * sizeof(float);
    static const int OUTPUT_SIZE = 1 * 300 * 6 * sizeof(float);
    static const int MAX_OUTPUT_DETECTIONS = 300;
    const char* INPUT_BLOB_NAME = "images";
    const char* OUTPUT_BLOB_NAME = "output0";

    size_t d_input_capacity = 0;
};

YOLODetector::YOLODetector(std::string engine_file_path) {

    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[ERROR]: Unable to open file: " << engine_file_path << std::endl;
        exit(1);
    }

    size_t size;

    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineModelStream(size);
    file.read(engineModelStream.data(), size);
    file.close();

    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineModelStream.data(), size);
    context = engine->createExecutionContext();

    cudaMalloc(&input_mem, INPUT_SIZE);
    cudaMalloc(&output_mem, OUTPUT_SIZE);

    cudaStreamCreate(&stream);

    preprocess_op = std::make_unique<cvcuda::ResizeCropConvertReformat>();

    nvcv::TensorShape outShape{{1, 3, MODEL_H, MODEL_W}, "NCHW"};
    nvcv::TensorDataStridedCuda::Buffer outBuf;
    outBuf.basePtr = static_cast<NVCVByte*>(input_mem);
    outBuf.strides[3] = sizeof(float);
    outBuf.strides[2] = MODEL_W * outBuf.strides[3];
    outBuf.strides[1] = MODEL_H * outBuf.strides[2];
    outBuf.strides[0] = 3 * outBuf.strides[1];
    nvcv::TensorDataStridedCuda outData(outShape, nvcv::DataType{NVCV_DATA_TYPE_F32}, outBuf);
    output_tensor = nvcv::TensorWrapData(outData);
}

YOLODetector::~YOLODetector() {
    if (stream) cudaStreamSynchronize(stream);
    if (d_input_u8) cudaFree(d_input_u8);
    cudaFree(input_mem);
    cudaFree(output_mem);

    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;
}

void YOLODetector::preprocessCudaToInputMem(const cv::Mat& img)
{
    cv::Mat host = img.isContinuous() ? img : img.clone();
    CV_Assert(host.type() == CV_8UC3);

    int w = host.cols, h = host.rows;
    size_t bytes = (size_t)w * h * 3;

    // (Re)alloc GPU staging buffer if needed
    if (bytes > d_input_capacity) {
        if (d_input_u8) cudaFree(d_input_u8);
        cudaMalloc(&d_input_u8, bytes);
        d_input_capacity = bytes;
    }

    // Upload image to GPU staging buffer
    cudaMemcpyAsync(d_input_u8, host.data, bytes, cudaMemcpyHostToDevice, stream);

    // Wrap staging buffer as NVCV input tensor: NHWC U8
    {
        nvcv::TensorShape inShape{{1, h, w, 3}, "NHWC"};
        nvcv::TensorDataStridedCuda::Buffer inBuf;
        inBuf.basePtr = static_cast<NVCVByte*>(d_input_u8);

        // Strides in bytes for NHWC:
        inBuf.strides[3] = 1;                 // C stride
        inBuf.strides[2] = 3;                 // W stride (3 bytes per pixel)
        inBuf.strides[1] = w * inBuf.strides[2]; // H stride
        inBuf.strides[0] = h * inBuf.strides[1]; // N stride

        nvcv::TensorDataStridedCuda inData(inShape, nvcv::DataType{NVCV_DATA_TYPE_U8}, inBuf);
        input_tensor = nvcv::TensorWrapData(inData);
    }

    // Run CV-CUDA op: resize + channel reverse + normalize + layout convert
    (*preprocess_op)(
        stream,
        input_tensor,
        output_tensor,
        {MODEL_W, MODEL_H},
        NVCV_INTERP_LINEAR,
        {0, 0},                 // crop offset
        NVCV_CHANNEL_REVERSE,   // BGR -> RGB (since OpenCV is BGR)
        1.0f / 255.0f,
        0.0f,
        false
    );
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& img, float threshold) {

    preprocessCudaToInputMem(img);                 // writes directly into input_mem
    std::vector<float> output(MAX_OUTPUT_DETECTIONS * 6);

    context->setTensorAddress(INPUT_BLOB_NAME, input_mem);
    context->setTensorAddress(OUTPUT_BLOB_NAME, output_mem);
    context->enqueueV3(stream);

    cudaMemcpyAsync(output.data(), output_mem, OUTPUT_SIZE, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    std::vector<Detection> results;

    for (int i = 0; i < MAX_OUTPUT_DETECTIONS; i++) {
        
        int offset = i * 6;

        float conf = output[offset+4];

        if (conf < threshold) break;
        
        float x = output[offset+0];
        float y = output[offset+1];
        float w = output[offset+2];
        float h = output[offset+3];

        int label = (int)output[offset+5];

        Detection det;
        det.rect = cv::Rect_<float>(x, y, w, h);
        det.prob = conf;
        det.label = label;

        results.push_back(det);
    }

    return results;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "[USAGE ERROR]: ./inference <engine_path> <image_path>" << std::endl;
        return 1;
    }

    std::string engine_path = argv[1];
    std::string image_path = argv[2];
    
    YOLODetector yolo(engine_path);
    cv::Mat img = cv::imread(image_path);

    if (img.empty()) {
        std::cerr << "[ERROR]: image empty at path: " << image_path << std::endl;
        return 1;
    }

    std::cout << "[INFO] Starting Warm-up" << std::endl;
    const int NUM_WARMUP = 10;
    for (int i = 0; i < NUM_WARMUP; ++i) {
       yolo.detect(img, 0.7f); 
    }

    const int NUM_ITERATIONS = 1000;
    std::cout << "[INFO] Starting benchmarking on " << NUM_ITERATIONS << " runs" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
       yolo.detect(img, 0.7f); 
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    double latency = (double)duration / NUM_ITERATIONS;
    double fps = 1000.0f / latency;

    std::cout << "Average latency: " << latency << " ms." << std::endl;
    std::cout << "Average fps: " << fps << " fps." << std::endl; 

    return 0;
}