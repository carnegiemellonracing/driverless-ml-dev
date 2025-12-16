#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

using namespace nvinfer1;

struct Detection
{
    cv::Rect_<float> rect;
    float prob;
    int label;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) override {
        if (severity <= Severity::kINFO) {
            std::cout << msg << "n";
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

    Logger logger;
    ICudaEngine* engine;
    IRuntime* runtime;
    IExecutionContext* context;
    cudaStream_t stream = nullptr;

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
    file.read(engineModelStream.data(), size)
    file.close;

    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineModelStream.data(), size);
    context = engine->createExecutionContext();

    if (!context->setInputShape()) {
        std::cerr << "[ERROR]: Unable to set input shape"<< std::endl;
    }

    cudaMalloc(&input_mem, INPUT_SIZE);
    cudaMalloc(&output_mem, OUTPUT_SIZE);

    cudaStreamCreate(&stream);
}

YOLODetector::~YOLODetector() {
    cudaFree(input_mem);
    cudaFree(output_mem);

    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;
}

std::vector<float> YOLO::preprocess(cv::Mat& img) {

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(640, 640));

    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
    
    cv::Mat transposed;
    std::vector<int> order = {2, 0, 1};
    cv::transposeND(resized, order, transposed);

    cv::Mat flat = transposed.isContinuous() ? transposed : transposed.clone();
    float* ptr = (float *)flat.data;

    size_t cnt = flat.total() * flat.channels();

    std::vector<float> result(640 * 640 * 3);
    result.assign(ptr, ptr + cnt);

    return result;
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& img, float threshold) {

    std::vector<float> input = preprocess(img)
    std::vector<float> output(MAX_OUTPUT_DETECTIONS * 6);

    cudaMemcpyAsync(input_mem, input.data(), INPUT_SIZE, cudaMemcpyHostToDevice, stream);
    context->setTensorAddress(INPUT_BLOB_NAME, input_mem);
    context->setTensorAddress(OUTPUT_BLOB_NAME, output_mem);
    context->enqueueV3(stream)
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

        int label = (int)output[offset+5]

        Detections det;
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

    if (!img.empty()) {
        auto dets = yolo.detect(img, 0.7f);
        std::cout << "Detected " << dets.size() << "objects." << std::endl;

        // TODO: Add an optional drawing of each bounding box
    }

    return 0;
}