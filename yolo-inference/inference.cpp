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

//cvcuda headers
#include <nvcv/Tensor.hpp>
#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/TensorDataAccess.hpp> //..?

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
    // CHANGED: No longer returns vector, preprocesses directly to GPU
    void preprocess_gpu(const cv::Mat& img);

    Logger logger;
    ICudaEngine* engine;
    IRuntime* runtime;
    IExecutionContext* context;
    cudaStream_t stream = nullptr;

    // ADD: CV-CUDA operator handle ..?
    NVCVOperatorHandle preprocess_op = nullptr

    void* input_mem = nullptr;
    void* output_mem = nullptr;

    //
    std::unique_ptr<cvcuda::ResizeCropConvertReformat> preprocess_op;
    nvcv::Tensor input_tensor;
    nvcv::Tensor output_tensor;

    void* input_image_gpu = nullptr;

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

    //NEW CVCUDA
    // Create the operator (C++ object, not a handle!)
    preprocess_op = std::make_unique<cvcuda::ResizeCropConvertReformat>();

    // Allocate GPU memory for input image
    const int max_input_width = 1920;
    const int max_input_height = 1080;
    cudaMalloc(&input_image_gpu, max_input_width * max_input_height * 3);

    // Create output tensor that wraps TensorRT's input buffer
    // Shape: {1, 3, 640, 640}, Layout: NCHW, Type: float32
    nvcv::TensorDataStridedCuda::Buffer output_buffer;
    output_buffer.basePtr = static_cast<NVCVByte*>(input_mem);
    output_buffer.strides[0] = 3 * 640 * 640 * sizeof(float);  // batch stride
    output_buffer.strides[1] = 640 * 640 * sizeof(float);      // channel stride
    output_buffer.strides[2] = 640 * sizeof(float);            // height stride
    output_buffer.strides[3] = sizeof(float);                  // width stride

    nvcv::TensorShape output_shape{{1, 3, 640, 640}, "NCHW"};

    nvcv::TensorDataStridedCuda output_data(
        output_shape,
        nvcv::DataType{NVCV_DATA_TYPE_F32},
        output_buffer
    );
    
    output_tensor = nvcv::TensorWrapData(output_data);

    std::cout << "[INFO]: CV-CUDA preprocessing initialized" << std::endl;
}

YOLODetector::~YOLODetector() {
    cudaFree(input_mem);
    cudaFree(output_mem);
    cudaFree(input_image_gpu); //NEW

    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;
}

std::vector<float> YOLODetector::preprocess_gpu(const cv::Mat& img) { //NEW

    nvtx3::scoped_range r{"preprocess_gpu"};
    
    // Copy input image to GPU
    {
        nvtx3::scoped_range r2{"H2D_image"};
        size_t img_size = img.rows * img.cols * 3;
        cudaMemcpyAsync(input_image_gpu, img.data, img_size, 
                       cudaMemcpyHostToDevice, stream);
    }
    
    // Wrap input image as CV-CUDA tensor
    // Shape: {1, height, width, 3}, Layout: NHWC, Type: uint8
    {
        nvcv::TensorDataStridedCuda::Buffer input_buffer;
        input_buffer.basePtr = static_cast<NVCVByte*>(input_image_gpu);
        input_buffer.strides[0] = img.rows * img.cols * 3;  // batch stride
        input_buffer.strides[1] = img.cols * 3;             // height stride
        input_buffer.strides[2] = 3;                        // width stride
        input_buffer.strides[3] = 1;                        // channel stride

        nvcv::TensorShape input_shape{{1, img.rows, img.cols, 3}, "NHWC"};
        
        nvcv::TensorDataStridedCuda input_data(
            input_shape,
            nvcv::DataType{NVCV_DATA_TYPE_U8},
            input_buffer
        );
        input_tensor = nvcv::TensorWrapData(input_data);
    }
    // Run fused preprocessing on GPU
    {
        nvtx3::scoped_range r2{"fused_preprocess"};
        
        // All operations in one call!
        (*preprocess_op)(
            stream,
            input_tensor,       // Input: BGR uint8 NHWC
            output_tensor,      // Output: RGB float32 NCHW
            {640, 640},         // resize_dim
            NVCV_INTERP_LINEAR, // interpolation
            {0, 0, 640, 640},   // crop_rect (x, y, w, h)
            NVCV_CHANNEL_REVERSE,  // BGR → RGB
            1.0f / 255.0f,      // scale
            0.0f                // offset
        );
    }
    
    // Data is now in input_mem (TensorRT buffer), ready for inference!
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& img, float threshold) {

    // std::vector<float> input;
    // {
    //     nvtx3::scoped_range r{"preprocess"};
    //     input = preprocess(img); 
    // }
    
    // GPU preprocessing
    preprocess_gpu(img);

    std::vector<float> output(MAX_OUTPUT_DETECTIONS * 6);
    
    // {
    //     nvtx3::scoped_range r{"H2D_memcpy"};
    //     cudaMemcpyAsync(input_mem, input.data(), INPUT_SIZE, cudaMemcpyHostToDevice, stream);
    // }
    
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
    for (int i = 0; i < NUM_WARMUP; ++i) {
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
