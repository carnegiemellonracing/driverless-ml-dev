#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <nvtx3/nvtx3.hpp>

#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/Tensor.hpp>

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
    std::vector<float> preprocess(const cv::Mat& img);
    void preprocessCuda(const cv::Mat& img);

    Logger logger;
    ICudaEngine* engine;
    IRuntime* runtime;
    IExecutionContext* context;
    cudaStream_t stream = nullptr;

    std::unique_ptr<cvcuda::ResizeCropConvertReformat> preprocess_op;

    nvcv::Tensor input_tensor;
    nvcv::Tensor output_tensor;

    void* input_img = nullptr;

    void* input_mem = nullptr;
    void* output_mem = nullptr;

    static const int INPUT_SIZE = 1 * 3 * 640 * 640 * sizeof(float);
    static const int OUTPUT_SIZE = 1 * 300 * 6 * sizeof(float);
    static const int MAX_OUTPUT_DETECTIONS = 300;
    const char* INPUT_BLOB_NAME = "images";
    const char* OUTPUT_BLOB_NAME = "output0";
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

    const int MAX_INPUT_WIDTH = 1920;
    const int MAX_INPUT_HEIGHT = 1080;
    cudaMalloc(&input_img, MAX_INPUT_WIDTH * MAX_INPUT_HEIGHT * 3);

    /* Wrap TensorRT's 'input_mem' as CV-CUDA output tensor */
    nvcv::TensorShape outputShape{{1, 3, 640, 640}, NVCV_TENSOR_NCHW};

    nvcv::TensorDataStridedCuda::Buffer outBuffer;
    outBuffer.basePtr = static_cast<NVCVByte*>(input_mem);
    outBuffer.strides[0] = 3 * 640 * 640 * sizeof(float);
    outBuffer.strides[1] = 640 * 640 * sizeof(float);
    outBuffer.strides[2] = 640 * sizeof(float);
    outBuffer.strides[3] = sizeof(float);

    nvcv::TensorDataStridedCuda outTensorData(
        outputShape,
        nvcv::DataType{NVCV_DATA_TYPE_F32},
        outBuffer
    );

    output_tensor = nvcv::TensorWrapData(outTensorData);
}

YOLODetector::~YOLODetector() {
    cudaFree(input_mem);
    cudaFree(output_mem);
    cudaFree(input_img);

    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;
}

std::vector<float> YOLODetector::preprocess(const cv::Mat& img) {

    cv::Mat resized;
    {
        nvtx3::scoped_range r{"resize"};
        cv::resize(img, resized, cv::Size(640, 640));
    }

    {
        nvtx3::scoped_range r{"colorTransform"};
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    }

    {
        nvtx3::scoped_range r{"fpConvert"};
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
    }
    
    nvtx3::scoped_range r{"HWC->CHW"};
    std::vector<float> result(3 * 640 * 640);
    float* data = result.data();

    const float* ptr = (float*)resized.data;
    const int num_pixels = 640*640;

    for (int i = 0; i < num_pixels; ++i) {
        int offset = i * 3;

        data[i] = ptr[offset];
        data[num_pixels + i] = ptr[offset + 1];
        data[2 * num_pixels + i] = ptr[offset + 2];
    }

    return result;
}

void YOLODetector::preprocessCuda(const cv::Mat& img) {
    nvtx3::scoped_range r{"preprocess_gpu"};

    {
        nvtx3::scoped_range r2{"H2D_image"};
        cv::Mat host = img.isContinuous() ? img : img.clone();
        cudaMemcpyAsync(input_img, host.data, 
                       host.total() * host.elemSize(), 
                       cudaMemcpyHostToDevice, stream);
    }

    {
        nvtx3::scoped_range r2{"create_input_tensor"};
        
        nvcv::TensorShape inShape{{1, img.rows, img.cols, 3}, "NHWC"};
        
        nvcv::TensorDataStridedCuda::Buffer inBuffer;
        inBuffer.basePtr = static_cast<NVCVByte*>(input_img);
        inBuffer.strides[0] = img.rows * img.cols * 3;
        inBuffer.strides[1] = img.cols * 3;
        inBuffer.strides[2] = 3;
        inBuffer.strides[3] = 1;
        
        nvcv::TensorDataStridedCuda inTensorData(
            inShape,
            nvcv::DataType{NVCV_DATA_TYPE_U8},
            inBuffer
        );
        
        input_tensor = nvcv::TensorWrapData(inTensorData);
    }

    {
        nvtx3::scoped_range r2{"fused_preprocess_kernel"};

        (*preprocess_op)(
            stream,
            input_tensor,
            output_tensor,
            {640, 640},
            NVCV_INTERP_LINEAR,
            {0, 0},
            NVCV_CHANNEL_REVERSE,
            1.0f / 255.0f,
            0.0f,
            false
        );
    }
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& img, float threshold) {
    
    /*
    std::vector<float> input;
    {
        nvtx3::scoped_range r{"preprocess"};
        input = preprocess(img); 
    }
    */

    preprocessCuda(img);

    std::vector<float> output(MAX_OUTPUT_DETECTIONS * 6);
    
    /*
    {
        nvtx3::scoped_range r{"H2D_memcpy"};
        cudaMemcpyAsync(input_mem, input.data(), INPUT_SIZE, cudaMemcpyHostToDevice, stream);
    }
    */

    {
        nvtx3::scoped_range r{"inference"};
        context->setTensorAddress(INPUT_BLOB_NAME, input_mem);
        context->setTensorAddress(OUTPUT_BLOB_NAME, output_mem);
        context->enqueueV3(stream);
    }

    {   
        nvtx3::scoped_range r{"D2H_memcpy"};
        cudaMemcpyAsync(output.data(), output_mem, OUTPUT_SIZE, cudaMemcpyDeviceToHost, stream);
    }

    {
        nvtx3::scoped_range r{"sync"};
        cudaStreamSynchronize(stream);
    }
    
    nvtx3::scoped_range r{"postprocess"};
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
    }

    std::cout << "[INFO] Starting Warm-up" << std::endl;
    const int NUM_WARMUP = 5;
    for (int i = 0; i < NUM_WARMUP; ++i) { //create new std vector that holds output of detect, struct Detection: stores detection points, then compare to image (use image loader, iou map) use include for header file 
       yolo.detect(img, 0.7f); 
    }

    const int NUM_ITERATIONS = 20;
    std::cout << "[INFO] Starting benchmarking on " << NUM_ITERATIONS << " runs" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
       yolo.detect(img, 0.7f); 
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int latency = duration / NUM_ITERATIONS;
    int fps = 1000.0f / latency;

    std::cout << "Average latency: " << latency << " ms." << std::endl;
    std::cout << "Average fps: " << fps << " fps." << std::endl; 

    return 0;
}