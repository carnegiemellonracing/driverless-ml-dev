#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef USE_TENSORRT
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <nvtx3/nvtx3.hpp>

#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/Tensor.hpp>

using namespace nvinfer1;
#endif

// ======================= Common =======================
struct Detection {
    cv::Rect_<float> rect; // x,y,w,h in 640x640 model space
    float prob = 0.0f;
    int label = -1;
};

class IDetector {
public:
    virtual ~IDetector() = default;
    virtual std::vector<Detection> detect(const cv::Mat& img_bgr, float conf_threshold) = 0;
};

static constexpr int MODEL_W = 640;
static constexpr int MODEL_H = 640;
static constexpr int NUM_CLASSES = 5;

#ifdef USE_TENSORRT
// ======================= TensorRT Logger =======================
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << "\n";
        }
    }
};

// ======================= TensorRT Detector =======================
class TRTDetector : public IDetector {
public:
    explicit TRTDetector(const std::string& engine_file_path);
    ~TRTDetector() override;

    std::vector<Detection> detect(const cv::Mat& img_bgr, float conf_threshold) override;

private:
    void preprocessCudaToInputMem(const cv::Mat& img_bgr);

    Logger logger_;
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    std::unique_ptr<cvcuda::ResizeCropConvertReformat> preprocess_op_;

    nvcv::Tensor input_tensor_;
    nvcv::Tensor output_tensor_;

    void* input_img_ = nullptr;
    void* input_mem_ = nullptr;
    void* output_mem_ = nullptr;

    int last_input_width_ = 0;
    int last_input_height_ = 0;

    static constexpr int MAX_OUTPUT_DETECTIONS = 300;
    static constexpr int INPUT_SIZE_BYTES  = 1 * 3 * MODEL_H * MODEL_W * sizeof(float);
    static constexpr int OUTPUT_SIZE_BYTES = 1 * MAX_OUTPUT_DETECTIONS * 6 * sizeof(float);

    const char* INPUT_BLOB_NAME_  = "images";
    const char* OUTPUT_BLOB_NAME_ = "output0";
};

TRTDetector::TRTDetector(const std::string& engine_file_path) {
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("[ERROR] Unable to open engine: " + engine_file_path);
    }

    file.seekg(0, file.end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, file.beg);

    std::vector<char> engine_model_stream(size);
    file.read(engine_model_stream.data(), size);
    file.close();

    runtime_ = createInferRuntime(logger_);
    if (!runtime_) {
        throw std::runtime_error("[ERROR] createInferRuntime failed");
    }

    engine_ = runtime_->deserializeCudaEngine(engine_model_stream.data(), size);
    if (!engine_) {
        throw std::runtime_error("[ERROR] deserializeCudaEngine failed");
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("[ERROR] createExecutionContext failed");
    }

    cudaMalloc(&input_mem_, INPUT_SIZE_BYTES);
    cudaMalloc(&output_mem_, OUTPUT_SIZE_BYTES);
    cudaStreamCreate(&stream_);

    preprocess_op_ = std::make_unique<cvcuda::ResizeCropConvertReformat>();

    const int MAX_INPUT_WIDTH  = 1920;
    const int MAX_INPUT_HEIGHT = 1080;
    cudaMalloc(&input_img_, MAX_INPUT_WIDTH * MAX_INPUT_HEIGHT * 3);

    nvcv::TensorShape out_shape{{1, 3, MODEL_H, MODEL_W}, NVCV_TENSOR_NCHW};
    nvcv::TensorDataStridedCuda::Buffer out_buffer;
    out_buffer.basePtr = static_cast<NVCVByte*>(input_mem_);
    out_buffer.strides[0] = 3 * MODEL_H * MODEL_W * sizeof(float);
    out_buffer.strides[1] = MODEL_H * MODEL_W * sizeof(float);
    out_buffer.strides[2] = MODEL_W * sizeof(float);
    out_buffer.strides[3] = sizeof(float);

    nvcv::TensorDataStridedCuda out_tensor_data(
        out_shape,
        nvcv::DataType{NVCV_DATA_TYPE_F32},
        out_buffer
    );
    output_tensor_ = nvcv::TensorWrapData(out_tensor_data);
}

TRTDetector::~TRTDetector() {
    if (input_mem_) cudaFree(input_mem_);
    if (output_mem_) cudaFree(output_mem_);
    if (input_img_) cudaFree(input_img_);
    if (stream_) cudaStreamDestroy(stream_);

    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

void TRTDetector::preprocessCudaToInputMem(const cv::Mat& img_bgr) {
    nvtx3::scoped_range r{"preprocess_gpu"};

    cv::Mat host = img_bgr.isContinuous() ? img_bgr : img_bgr.clone();

    const int MAX_INPUT_WIDTH  = 1920;
    const int MAX_INPUT_HEIGHT = 1080;
    if (host.cols > MAX_INPUT_WIDTH || host.rows > MAX_INPUT_HEIGHT) {
        float scale = std::min(
            static_cast<float>(MAX_INPUT_WIDTH) / host.cols,
            static_cast<float>(MAX_INPUT_HEIGHT) / host.rows
        );
        cv::resize(host, host, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    cudaMemcpyAsync(
        input_img_,
        host.data,
        host.total() * host.elemSize(),
        cudaMemcpyHostToDevice,
        stream_
    );

    last_input_width_ = host.cols;
    last_input_height_ = host.rows;

    nvcv::TensorShape in_shape{{1, last_input_height_, last_input_width_, 3}, "NHWC"};
    nvcv::TensorDataStridedCuda::Buffer in_buffer;
    in_buffer.basePtr = static_cast<NVCVByte*>(input_img_);
    in_buffer.strides[0] = last_input_height_ * last_input_width_ * 3;
    in_buffer.strides[1] = last_input_width_ * 3;
    in_buffer.strides[2] = 3;
    in_buffer.strides[3] = 1;

    nvcv::TensorDataStridedCuda in_tensor_data(
        in_shape,
        nvcv::DataType{NVCV_DATA_TYPE_U8},
        in_buffer
    );
    input_tensor_ = nvcv::TensorWrapData(in_tensor_data);

    (*preprocess_op_)(
        stream_,
        input_tensor_,
        output_tensor_,
        {MODEL_W, MODEL_H},
        NVCV_INTERP_LINEAR,
        {0, 0},
        NVCV_CHANNEL_REVERSE, // BGR -> RGB
        1.0f / 255.0f,
        0.0f,
        false
    );
}

std::vector<Detection> TRTDetector::detect(const cv::Mat& img_bgr, float conf_threshold) {
    preprocessCudaToInputMem(img_bgr);

    std::vector<float> output(MAX_OUTPUT_DETECTIONS * 6);

    {
        nvtx3::scoped_range r{"inference"};
        context_->setTensorAddress(INPUT_BLOB_NAME_, input_mem_);
        context_->setTensorAddress(OUTPUT_BLOB_NAME_, output_mem_);
        context_->enqueueV3(stream_);
    }

    {
        nvtx3::scoped_range r{"D2H"};
        cudaMemcpyAsync(
            output.data(),
            output_mem_,
            OUTPUT_SIZE_BYTES,
            cudaMemcpyDeviceToHost,
            stream_
        );
        cudaStreamSynchronize(stream_);
    }

    std::vector<Detection> results;
    results.reserve(64);

    for (int i = 0; i < MAX_OUTPUT_DETECTIONS; i++) {
        int off = i * 6;

        float x1   = output[off + 0];
        float y1   = output[off + 1];
        float x2   = output[off + 2];
        float y2   = output[off + 3];
        float conf = output[off + 4];
        int label  = static_cast<int>(output[off + 5]);

        if (conf < conf_threshold) continue;

        float w = x2 - x1;
        float h = y2 - y1;
        if (w <= 1.0f || h <= 1.0f) continue;
        if (label < 0 || label >= NUM_CLASSES) continue;

        Detection d;
        d.rect = cv::Rect_<float>(x1, y1, w, h);
        d.prob = conf;
        d.label = label;
        results.push_back(d);
    }

    return results;
}
#endif

#ifdef USE_ONNX
// ======================= ONNX Detector =======================
class ONNXDetector : public IDetector {
public:
    explicit ONNXDetector(const std::string& onnx_file_path) {
        net_ = cv::dnn::readNetFromONNX(onnx_file_path);
        if (net_.empty()) {
            throw std::runtime_error("[ERROR] Failed to load ONNX model: " + onnx_file_path);
        }

        try {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } catch (...) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    }

    std::vector<Detection> detect(const cv::Mat& img_bgr, float conf_threshold) override {
        cv::Mat blob;
        cv::dnn::blobFromImage(
            img_bgr,
            blob,
            1.0 / 255.0,
            cv::Size(MODEL_W, MODEL_H),
            cv::Scalar(),
            true,   // BGR -> RGB
            false
        );

        net_.setInput(blob);

        std::vector<cv::Mat> outs;
        net_.forward(outs, net_.getUnconnectedOutLayersNames());
        if (outs.empty()) return {};

        cv::Mat det = normalizeOutput(outs[0]);

        // Already post-NMS: [N,6] = x1,y1,x2,y2,conf,cls
        if (det.cols == 6) {
            std::vector<Detection> results;
            results.reserve(det.rows);

            for (int i = 0; i < det.rows; i++) {
                const float* row = det.ptr<float>(i);

                float x1   = row[0];
                float y1   = row[1];
                float x2   = row[2];
                float y2   = row[3];
                float conf = row[4];
                int label  = static_cast<int>(row[5]);

                if (conf < conf_threshold) continue;

                float w = x2 - x1;
                float h = y2 - y1;
                if (w <= 1.0f || h <= 1.0f) continue;
                if (label < 0 || label >= NUM_CLASSES) continue;

                Detection d;
                d.rect = cv::Rect_<float>(x1, y1, w, h);
                d.prob = conf;
                d.label = label;
                results.push_back(d);
            }

            return results;
        }

        // Raw YOLO output:
        //   Nx(5+C): cx,cy,w,h,obj,cls...
        //   Nx(4+C): cx,cy,w,h,cls...
        const bool has_objectness = (det.cols == 5 + NUM_CLASSES);
        const bool no_objectness  = (det.cols == 4 + NUM_CLASSES);

        if (!has_objectness && !no_objectness) {
            throw std::runtime_error(
                "[ERROR] Unexpected ONNX output shape. Expected Nx6, Nx" +
                std::to_string(5 + NUM_CLASSES) + ", or Nx" +
                std::to_string(4 + NUM_CLASSES) + ". Got Nx" +
                std::to_string(det.cols)
            );
        }

        std::vector<Detection> raw;
        raw.reserve(det.rows);

        for (int i = 0; i < det.rows; i++) {
            const float* row = det.ptr<float>(i);

            float cx = row[0];
            float cy = row[1];
            float w  = row[2];
            float h  = row[3];

            float obj = has_objectness ? row[4] : 1.0f;
            int cls_start = has_objectness ? 5 : 4;

            int best_cls = -1;
            float best_cls_prob = 0.0f;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float p = row[cls_start + c];
                if (p > best_cls_prob) {
                    best_cls_prob = p;
                    best_cls = c;
                }
            }

            float score = obj * best_cls_prob;
            if (score < conf_threshold) continue;
            if (best_cls < 0 || best_cls >= NUM_CLASSES) continue;
            if (w <= 1.0f || h <= 1.0f) continue;

            Detection d;
            d.rect = cv::Rect_<float>(cx - 0.5f * w, cy - 0.5f * h, w, h);
            d.prob = score;
            d.label = best_cls;
            raw.push_back(d);
        }

        return classWiseNMS(raw, conf_threshold, 0.45f);
    }

private:
    cv::Mat normalizeOutput(const cv::Mat& out) const {
        // Supports:
        // [N, A]
        // [1, N, A]
        // [1, A, N] -> transpose to [N, A]
        if (out.dims == 2) {
            return out;
        }

        if (out.dims == 3 && out.size[0] == 1) {
            const int d1 = out.size[1];
            const int d2 = out.size[2];
            float* data = reinterpret_cast<float*>(out.data);

            if (d2 == 6 || d2 == 5 + NUM_CLASSES || d2 == 4 + NUM_CLASSES) {
                return cv::Mat(d1, d2, CV_32F, data);
            }

            if (d1 == 6 || d1 == 5 + NUM_CLASSES || d1 == 4 + NUM_CLASSES) {
                cv::Mat tmp(d1, d2, CV_32F, data);
                cv::Mat transposed;
                cv::transpose(tmp, transposed);
                return transposed;
            }
        }

        throw std::runtime_error("[ERROR] Unsupported ONNX output dims");
    }

    std::vector<Detection> classWiseNMS(
        const std::vector<Detection>& dets,
        float score_threshold,
        float nms_threshold
    ) const {
        std::vector<Detection> final_dets;

        for (int c = 0; c < NUM_CLASSES; c++) {
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            std::vector<Detection> class_dets;

            for (const auto& d : dets) {
                if (d.label != c) continue;

                boxes.emplace_back(
                    static_cast<int>(std::round(d.rect.x)),
                    static_cast<int>(std::round(d.rect.y)),
                    static_cast<int>(std::round(d.rect.width)),
                    static_cast<int>(std::round(d.rect.height))
                );
                scores.push_back(d.prob);
                class_dets.push_back(d);
            }

            std::vector<int> keep;
            cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, keep);

            for (int idx : keep) {
                final_dets.push_back(class_dets[idx]);
            }
        }

        return final_dets;
    }

    cv::dnn::Net net_;
};
#endif

static std::unique_ptr<IDetector> makeDetector(const std::string& model_path) {
#ifdef USE_TENSORRT
    return std::make_unique<TRTDetector>(model_path);
#elif defined(USE_ONNX)
    return std::make_unique<ONNXDetector>(model_path);
#else
#error "Must compile with either -DUSE_TENSORRT or -DUSE_ONNX"
#endif
}

// ======================= CLI =======================
struct Args {
    std::string model_path;
    std::string image_path;

    float map_conf = 0.5f;
    int warmup = 5;
    int iters = 20;
};

static std::string getOpt(int argc, char** argv, const std::string& key, const std::string& def = "") {
    for (int i = 1; i + 1 < argc; i++) {
        if (key == argv[i]) return argv[i + 1];
    }
    return def;
}

static int getOptInt(int argc, char** argv, const std::string& key, int def) {
    auto s = getOpt(argc, argv, key, "");
    return s.empty() ? def : std::stoi(s);
}

static float getOptFloat(int argc, char** argv, const std::string& key, float def) {
    auto s = getOpt(argc, argv, key, "");
    return s.empty() ? def : std::stof(s);
}

// ======================= Benchmark =======================
void runBench(const Args& a) {
    auto detector = makeDetector(a.model_path);

    cv::Mat img = cv::imread(a.image_path);
    if (img.empty()) {
        throw std::runtime_error("[ERROR] image empty at path: " + a.image_path);
    }

    std::cout << "[INFO] Model:  " << a.model_path << "\n";
    std::cout << "[INFO] Image:  " << a.image_path << "\n";
    std::cout << "[INFO] Conf:   " << a.map_conf << "\n";
    std::cout << "[INFO] Warmup: " << a.warmup << "\n";
    std::cout << "[INFO] Iters:  " << a.iters << "\n";

    std::cout << "[INFO] Starting warm-up...\n";
    for (int i = 0; i < a.warmup; ++i) {
        auto preds = detector->detect(img, a.map_conf);
        (void)preds;
    }

    std::cout << "[INFO] Starting benchmark...\n";

    std::vector<Detection> last_preds;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < a.iters; ++i) {
        last_preds = detector->detect(img, a.map_conf);
    }

    auto end = std::chrono::high_resolution_clock::now();

    double total_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    double latency_ms = total_ms / static_cast<double>(a.iters);
    double fps = (latency_ms > 0.0) ? (1000.0 / latency_ms) : 0.0;

    std::cout << "Detections kept: " << last_preds.size() << "\n";
    std::cout << "Average latency: " << latency_ms << " ms\n";
    std::cout << "Average FPS:     " << fps << "\n";
}

// ======================= main =======================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr
            << "[USAGE]\n"
            << "  ./inference <model_path> <image_path> "
            << "[--map_conf 0.5] [--warmup 5] [--iters 20]\n";
        return 1;
    }

    Args a;
    a.model_path = argv[1];
    a.image_path = argv[2];
    a.map_conf = getOptFloat(argc, argv, "--map_conf", a.map_conf);
    a.warmup = getOptInt(argc, argv, "--warmup", a.warmup);
    a.iters = getOptInt(argc, argv, "--iters", a.iters);

    try {
        runBench(a);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    return 0;
}