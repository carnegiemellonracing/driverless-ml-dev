#include <iostream>
#include <vector>
#include <string>
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

    std::vector<void*> buffers;

    static const int INPUT_H = 640;
    static const int INPUT_W = 640;
    static const int OUTPUT_SIZE = 1 * 300 * 6;
}
    // init: Load Engine, Allocate GPU Memory

    // preprocess: image -> flat vector

    // detect: main inference pipeline

int main() {
    return 1;
}