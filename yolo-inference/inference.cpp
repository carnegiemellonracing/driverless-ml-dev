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

    // Track input dimensions to detect changes
    int last_input_width = 0;
    int last_input_height = 0;

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
        
        // Bounds check - resize if image exceeds max buffer size
        const int MAX_INPUT_WIDTH = 1920;
        const int MAX_INPUT_HEIGHT = 1080;
        cv::Mat host = img.isContinuous() ? img : img.clone();
        
        if (host.cols > MAX_INPUT_WIDTH || host.rows > MAX_INPUT_HEIGHT) {
            // Calculate scale to fit within bounds while maintaining aspect ratio
            float scale = std::min(
                static_cast<float>(MAX_INPUT_WIDTH) / host.cols,
                static_cast<float>(MAX_INPUT_HEIGHT) / host.rows
            );
            cv::resize(host, host, cv::Size(), scale, scale, cv::INTER_LINEAR);
        }
        
        cudaMemcpyAsync(input_img, host.data, 
                       host.total() * host.elemSize(), 
                       cudaMemcpyHostToDevice, stream);
        
        // Update dimensions after potential resize
        last_input_width = host.cols;
        last_input_height = host.rows;
    }

    {
        nvtx3::scoped_range r2{"create_input_tensor"};
        
        // Use tracked dimensions (which account for any resize that occurred)
        nvcv::TensorShape inShape{{1, last_input_height, last_input_width, 3}, "NHWC"};
        
        nvcv::TensorDataStridedCuda::Buffer inBuffer;
        inBuffer.basePtr = static_cast<NVCVByte*>(input_img);
        inBuffer.strides[0] = last_input_height * last_input_width * 3;
        inBuffer.strides[1] = last_input_width * 3;
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

    // Synchronize to ensure preprocessing is complete before inference
    cudaStreamSynchronize(stream);
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

        if (conf < threshold) continue;
        
        // YOLO/TensorRT outputs boxes as (x1, y1, x2, y2) corner coordinates
        float x1 = output[offset+0];
        float y1 = output[offset+1];
        float x2 = output[offset+2];
        float y2 = output[offset+3];
        
        // Convert to (x, y, w, h) format for cv::Rect
        float x = x1;
        float y = y1;
        float w = x2 - x1;
        float h = y2 - y1;

        int label = (int)output[offset+5];

        Detection det;
        det.rect = cv::Rect_<float>(x, y, w, h);
        det.prob = conf;
        det.label = label;

        results.push_back(det);
    }

    return results;
}

// ============= VALIDATION CODE =============

// Calculate IoU between two bounding boxes
float calculateIoU(const cv::Rect_<float>& a, const cv::Rect_<float>& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float unionArea = areaA + areaB - intersection;
    
    return (unionArea > 0) ? (intersection / unionArea) : 0.0f;
}

// Parse YOLO format label file: class x_center y_center width height (normalized)
// Note: We use MODEL_INPUT_SIZE (640) because model predictions are in 640x640 space
std::vector<Detection> parseYOLOLabels(const std::string& label_path, int /*img_width*/, int /*img_height*/) {
    const int MODEL_INPUT_SIZE = 640;  // Model input resolution
    
    std::vector<Detection> gt;
    std::ifstream file(label_path);
    if (!file.is_open()) return gt;
    
    int cls;
    float x_center, y_center, w, h;
    while (file >> cls >> x_center >> y_center >> w >> h) {
        Detection d;
        // Convert normalized coords to 640x640 model input space
        float abs_w = w * MODEL_INPUT_SIZE;
        float abs_h = h * MODEL_INPUT_SIZE;
        float abs_x = x_center * MODEL_INPUT_SIZE - abs_w / 2.0f;
        float abs_y = y_center * MODEL_INPUT_SIZE - abs_h / 2.0f;
        d.rect = cv::Rect_<float>(abs_x, abs_y, abs_w, abs_h);
        d.label = cls;
        d.prob = 1.0f; // ground truth
        gt.push_back(d);
    }
    return gt;
}

// Get list of files in a directory with a specific extension
std::vector<std::string> getFilesInDir(const std::string& dir, const std::string& ext) {
    std::vector<std::string> files;
    cv::glob(dir + "/*" + ext, files, false);
    return files;
}

// Extract base filename without extension
std::string getBaseName(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    size_t lastDot = path.find_last_of(".");
    if (lastSlash == std::string::npos) lastSlash = 0;
    else lastSlash++;
    if (lastDot == std::string::npos || lastDot < lastSlash) lastDot = path.length();
    return path.substr(lastSlash, lastDot - lastSlash);
}

struct ValidationMetrics {
    int total_gt = 0;       // Total ground truth objects
    int total_pred = 0;     // Total predictions
    int true_positives = 0; // Correct detections
    
    float precision() const { return total_pred > 0 ? (float)true_positives / total_pred : 0.0f; }
    float recall() const { return total_gt > 0 ? (float)true_positives / total_gt : 0.0f; }
    float f1() const { 
        float p = precision(), r = recall();
        return (p + r > 0) ? (2.0f * p * r) / (p + r) : 0.0f;
    }
};

// Evaluate detections against ground truth for a single image
void evaluateImage(const std::vector<Detection>& preds, 
                   const std::vector<Detection>& gts,
                   float iou_threshold,
                   ValidationMetrics& metrics) {
    
    metrics.total_gt += gts.size();
    metrics.total_pred += preds.size();
    
    std::vector<bool> gt_matched(gts.size(), false);
    
    // For each prediction, find best matching GT (same class, highest IoU)
    for (const auto& pred : preds) {
        float best_iou = 0.0f;
        int best_idx = -1;
        
        for (size_t i = 0; i < gts.size(); i++) {
            if (gt_matched[i]) continue;  // Already matched
            if (gts[i].label != pred.label) continue;  // Different class
            
            float iou = calculateIoU(pred.rect, gts[i].rect);
            if (iou > best_iou && iou >= iou_threshold) {
                best_iou = iou;
                best_idx = i;
            }
        }
        
        if (best_idx >= 0) {
            gt_matched[best_idx] = true;
            metrics.true_positives++;
        }
    }
}

// FSOCO cone class names
const std::vector<std::string> CLASS_NAMES = {
    "unknown_cone",
    "yellow_cone",
    "blue_cone",
    "orange_cone",
    "large_orange_cone"
};

std::string getClassName(int label) {
    if (label >= 0 && label < (int)CLASS_NAMES.size()) {
        return CLASS_NAMES[label];
    }
    return "class_" + std::to_string(label);
}

void runValidation(const std::string& engine_path, 
                   const std::string& dataset_dir,
                   float conf_threshold = 0.5f,
                   float iou_threshold = 0.5f) {
    
    // FSOCO structure: dataset_dir/images/test/ and dataset_dir/labels/test/
    std::string images_dir = dataset_dir + "/images/test";
    std::string labels_dir = dataset_dir + "/labels/test";
    
    YOLODetector yolo(engine_path);
    
    std::vector<std::string> image_files = getFilesInDir(images_dir, ".jpg");
    std::vector<std::string> png_files = getFilesInDir(images_dir, ".png");
    image_files.insert(image_files.end(), png_files.begin(), png_files.end());
    
    if (image_files.empty()) {
        std::cerr << "[ERROR] No images found in: " << images_dir << std::endl;
        return;
    }
    
    ValidationMetrics overall;
    std::unordered_map<int, ValidationMetrics> per_class;
    
    std::cout << "[INFO] Images dir: " << images_dir << std::endl;
    std::cout << "[INFO] Labels dir: " << labels_dir << std::endl;
    std::cout << "[INFO] Validating on " << image_files.size() << " images..." << std::endl;
    
    int images_processed = 0;
    for (const auto& img_path : image_files) {
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) {
            std::cerr << "[WARN] Could not read: " << img_path << std::endl;
            continue;
        }
        
        // Find corresponding label file
        std::string base = getBaseName(img_path);
        std::string label_path = labels_dir + "/" + base + ".txt";
        
        std::vector<Detection> gt = parseYOLOLabels(label_path, img.cols, img.rows);
        std::vector<Detection> preds = yolo.detect(img, conf_threshold);
        
        // Evaluate this image - need a copy for per-class TP tracking
        std::vector<bool> gt_matched(gt.size(), false);
        
        overall.total_gt += gt.size();
        overall.total_pred += preds.size();
        
        for (const auto& pred : preds) {
            float best_iou = 0.0f;
            int best_idx = -1;
            
            for (size_t i = 0; i < gt.size(); i++) {
                if (gt_matched[i]) continue;
                if (gt[i].label != pred.label) continue;
                
                float iou = calculateIoU(pred.rect, gt[i].rect);
                if (iou > best_iou && iou >= iou_threshold) {
                    best_iou = iou;
                    best_idx = i;
                }
            }
            
            if (best_idx >= 0) {
                gt_matched[best_idx] = true;
                overall.true_positives++;
                per_class[pred.label].true_positives++;
            }
        }
        
        // Per-class ground truth and prediction counts
        for (const auto& g : gt) {
            per_class[g.label].total_gt++;
        }
        for (const auto& p : preds) {
            per_class[p.label].total_pred++;
        }
        
        images_processed++;
        if (images_processed % 100 == 0) {
            std::cout << "[INFO] Processed " << images_processed << "/" << image_files.size() << " images..." << std::endl;
        }
    }
    
    // Print results
    std::cout << "\n========== VALIDATION RESULTS ==========" << std::endl;
    std::cout << "Dataset:        " << dataset_dir << std::endl;
    std::cout << "Images:         " << images_processed << std::endl;
    std::cout << "IoU Threshold:  " << iou_threshold << std::endl;
    std::cout << "Conf Threshold: " << conf_threshold << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Total Ground Truth: " << overall.total_gt << std::endl;
    std::cout << "Total Predictions:  " << overall.total_pred << std::endl;
    std::cout << "True Positives:     " << overall.true_positives << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "Precision: " << (overall.precision() * 100.0f) << "%" << std::endl;
    std::cout << "Recall:    " << (overall.recall() * 100.0f) << "%" << std::endl;
    std::cout << "F1 Score:  " << (overall.f1() * 100.0f) << "%" << std::endl;
    
    // Print per-class metrics
    std::cout << "\n------------ Per-Class Metrics ----------" << std::endl;
    for (const auto& kv : per_class) {
        std::string name = getClassName(kv.first);
        const ValidationMetrics& m = kv.second;
        std::cout << name << " (class " << kv.first << "):" << std::endl;
        std::cout << "  GT: " << m.total_gt 
                  << " | Pred: " << m.total_pred 
                  << " | TP: " << m.true_positives 
                  << " | P: " << (m.precision() * 100.0f) << "%" 
                  << " | R: " << (m.recall() * 100.0f) << "%" << std::endl;
    }
    std::cout << "=========================================\n" << std::endl;
}


int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "[USAGE]:" << std::endl;
        std::cerr << "  Benchmark: ./inference <engine_path> <image_path>" << std::endl;
        std::cerr << "  Validate:  ./inference <engine_path> <dataset_dir> val [conf] [iou]" << std::endl;
        std::cerr << "             (dataset_dir should contain images/test and labels/test)" << std::endl;
        return 1;
    }

    std::string engine_path = argv[1];
    std::string path2 = argv[2];

    // Check if validation mode (3rd arg is "val" or ends with "/" indicating a directory)
    if (argc >= 4 && std::string(argv[3]) == "val") {
        float conf = (argc >= 5) ? std::stof(argv[4]) : 0.5f;
        float iou = (argc >= 6) ? std::stof(argv[5]) : 0.5f;
        
        runValidation(engine_path, path2, conf, iou);
        return 0;
    }

    // Original benchmark mode
    YOLODetector yolo(engine_path);
    cv::Mat img = cv::imread(path2);

    if (img.empty()) {
        std::cerr << "[ERROR]: image empty at path: " << path2 << std::endl;
        return 1;
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
}