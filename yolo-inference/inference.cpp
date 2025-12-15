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
        std::cout << msg << "n";
    }
} gLogger;

class YOLODetector {
public:
    YOLODetector(std::string engine_file_path);
    ~YOLODetector();
    std::vector<float> preprocess(const cv::Mat& img);
    std::vector<Detection> detect (const cv::Mat& img, float conf);

private:
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
        std::cerr << "[ERROR]: Unable to open file" << std::endl;
    }

    std::vector<char> engineModelStream(size);
    size_t size;

    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
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

    cudaStreamCreate(stream);
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

    cv::cvtColor(resized, resized, cv::COLORBGR2RGB);

    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);
    
    cv::Mat transposed;
    std::vector<int> order = {2, 0, 1};
    cv::transposeND(resized, order, transposed);

    cv::Mat flat = transposed.isContinuous() ? transposed : transposed.clone();
    float* ptr = (float *)flat.data;

    size_t cnt = flat.total() * flat.channels();

    std::vector<float> result;
    result.assign(ptr, ptr + cnt);

    return result;
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& img, float threshold) {

    std::vector<float> input = preprocess(img)
    std::vector<float> output(MAX_OUTPUT_DETECTIONS * 6);

    cudaMemcpyAsync(input_mem, input, INPUT_SIZE, cudaMemcpyHostToDevice, stream);
    context->setTensorAddress(INPUT_BLOB_NAME, input_mem);
    context->setTensorAddress(OUTPUT_BLOB_NAME, output_mem);
    context->enqueueV3(stream)
    cudaMemcpyAsync(output, output_mem, OUTPUT_SIZE, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    
    std::vector<Detection> results;

    for (char i = 0; i < MAX_OUTPUT_DETECTIONS; i++) {
        
        float conf = output[i+4];

        if (conf < threshold) break;
        
        float x = output[i+0];
        float y = output[i+1];
        float w = output[i+2];
        float h = output[i+3];

        int label = output[i+5]

        Detections det;
        det.rect = cv::Rect_<float>(x, y, w, h);
        det.prob = conf;
        det.label = label;

        results.push_back(det);
    }
}

int main(int argc, char **argv) {
    if (argc = 0) {
        std::cerr << "[ERROR]: include argument for engine file path" << std::endl;
        return 1;
    }

    YOLODetector yolo(argv[1]);

    if (argc = 1) {
        std::cerr << "[ERROR]: include argument for image file path" << std::endl;
        return 1;
    }

    cv::Mat img = cv::imread(argv[2]);

    if (!img.empty()) {
        auto dets = yolo.detect(img, 0.7f);
        std::cout << "Detected " << dets.size() << "objects." << std::endl;
    }

    return 0;
}