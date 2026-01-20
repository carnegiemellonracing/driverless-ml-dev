#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <nvtx3/nvtx3.hpp>

// CV-CUDA headers
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
    std::vector<Detection> detect(const cv::Mat& img, float conf);

private:
    void preprocess_gpu(const cv::Mat& img);

    Logger logger;
    ICudaEngine* engine;
    IRuntime* runtime;
    IExecutionContext* context;
    cudaStream_t stream = nullptr;

    void* input_mem = nullptr;   // TensorRT's input buffer
    void* output_mem = nullptr;  // TensorRT's output buffer

    // CV-CUDA operator
    std::unique_ptr<cvcuda::ResizeCropConvertReformat> preprocess_op;

    // CV-CUDA tensors
    nvcv::Tensor input_tensor;   // Will be created per-frame in preprocess_gpu()
    nvcv::Tensor output_tensor;  // Wraps input_mem (created once in constructor)

    void* input_image_gpu = nullptr;  // Temporary buffer for uploading images

    static const int INPUT_SIZE = 1 * 3 * 640 * 640 * sizeof(float);
    static const int OUTPUT_SIZE = 1 * 300 * 6 * sizeof(float);
    static const int MAX_OUTPUT_DETECTIONS = 300;
    const char* INPUT_BLOB_NAME = "images";
    const char* OUTPUT_BLOB_NAME = "output0";
};

YOLODetector::YOLODetector(std::string engine_file_path) {

    // ============================================
    // 1. TensorRT Setup
    // ============================================
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

    // ============================================
    // 2. Allocate GPU Buffers
    // ============================================
    cudaMalloc(&input_mem, INPUT_SIZE);    // TensorRT expects preprocessed data here
    cudaMalloc(&output_mem, OUTPUT_SIZE);  // TensorRT writes detections here
    cudaStreamCreate(&stream);

    // ============================================
    // 3. CV-CUDA Setup
    // ============================================
    
    // Create the fused preprocessing operator
    preprocess_op = std::make_unique<cvcuda::ResizeCropConvertReformat>();

    // Allocate temporary buffer for uploading raw camera images
    const int max_input_width = 1920;
    const int max_input_height = 1080;
    cudaMalloc(&input_image_gpu, max_input_width * max_input_height * 3);

    // ============================================
    // 4. Wrap TensorRT's input_mem as CV-CUDA output tensor
    // ============================================
    
    // This is the KEY part: we're telling CV-CUDA to write directly
    // into TensorRT's input buffer (input_mem) instead of allocating
    // its own memory
    
    nvcv::TensorShape outputShape{{1, 3, 640, 640}, "NCHW"};
    
    // Buffer descriptor tells CV-CUDA where the memory is and how to navigate it
    nvcv::TensorDataStridedCuda::Buffer outBuffer;
    outBuffer.basePtr = static_cast<NVCVByte*>(input_mem);  // Point to TensorRT's buffer!
    outBuffer.strides[0] = 3 * 640 * 640 * sizeof(float);   // Batch stride
    outBuffer.strides[1] = 640 * 640 * sizeof(float);       // Channel stride  
    outBuffer.strides[2] = 640 * sizeof(float);             // Row stride
    outBuffer.strides[3] = sizeof(float);                   // Column stride
    
    // Create the tensor data descriptor
    nvcv::TensorDataStridedCuda outTensorData(
        outputShape,
        nvcv::DataType{NVCV_DATA_TYPE_F32},
        outBuffer
    );
    
    // Wrap it as a tensor
    output_tensor = nvcv::TensorWrapData(outTensorData);
    
    // Now when CV-CUDA writes to output_tensor, it's actually writing to input_mem!

    std::cout << "[INFO]: CV-CUDA preprocessing initialized" << std::endl;
}

YOLODetector::~YOLODetector() {
    cudaFree(input_mem);
    cudaFree(output_mem);
    cudaFree(input_image_gpu);

    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;
}

void YOLODetector::preprocess_gpu(const cv::Mat& img) {
    nvtx3::scoped_range r{"preprocess_gpu"};
    
    // ============================================
    // Step 1: Upload raw image from CPU to GPU
    // ============================================
    {
        nvtx3::scoped_range r2{"H2D_image"};
        cudaMemcpyAsync(input_image_gpu, img.data, 
                       img.rows * img.cols * 3, 
                       cudaMemcpyHostToDevice, stream);
    }
    
    // ============================================
    // Step 2: Wrap uploaded image as CV-CUDA input tensor
    // ============================================
    {
        nvtx3::scoped_range r2{"create_input_tensor"};
        
        nvcv::TensorShape inShape{{1, img.rows, img.cols, 3}, "NHWC"};
        
        nvcv::TensorDataStridedCuda::Buffer inBuffer;
        inBuffer.basePtr = static_cast<NVCVByte*>(input_image_gpu);
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
    
    // ============================================
    // Step 3: Run fused preprocessing
    // ============================================
    {
        nvtx3::scoped_range r2{"fused_preprocess"};
        
        // This single operation does:
        // - Resize to 640x640
        // - BGR → RGB color conversion
        // - Normalize (/255)
        // - HWC → CHW layout conversion
        // - uint8 → float32 type conversion
        // All written directly to input_mem (via output_tensor)!
        
        (*preprocess_op)(
            stream,
            input_tensor,          // Input: BGR uint8 NHWC from camera
            output_tensor,         // Output: RGB float32 NCHW in input_mem
            {640, 640},            // Resize dimensions
            NVCV_INTERP_LINEAR,    // Bilinear interpolation
            {0, 0, 640, 640},      // Crop rect (no crop, use full image)
            NVCV_CHANNEL_REVERSE,  // BGR → RGB
            1.0f / 255.0f,         // Scale for normalization
            0.0f                   // Offset for normalization
        );
    }
    
    // Result is now in input_mem, ready for TensorRT!
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& img, float threshold) {

    // ============================================
    // Step 1: Preprocess on GPU
    // ============================================
    preprocess_gpu(img);  // Writes to input_mem

    std::vector<float> output(MAX_OUTPUT_DETECTIONS * 6);
    
    // ============================================
    // Step 2: Run TensorRT inference
    // ============================================
    {
        nvtx3::scoped_range r{"inference"};
        context->setTensorAddress(INPUT_BLOB_NAME, input_mem);   // Already has preprocessed data!
        context->setTensorAddress(OUTPUT_BLOB_NAME, output_mem);
        context->enqueueV3(stream);
    }

    // ============================================
    // Step 3: Copy results back to CPU
    // ============================================
    {   
        nvtx3::scoped_range r{"D2H_memcpy"};
        cudaMemcpyAsync(output.data(), output_mem, OUTPUT_SIZE, 
                       cudaMemcpyDeviceToHost, stream);
    }

    {
        nvtx3::scoped_range r{"sync"};
        cudaStreamSynchronize(stream);
    }
    
    // ============================================
    // Step 4: Parse detections
    // ============================================
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
