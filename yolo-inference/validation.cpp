#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <memory>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <nvtx3/nvtx3.hpp>

#include <cvcuda/OpResizeCropConvertReformat.hpp>
#include <nvcv/Tensor.hpp>

using namespace nvinfer1;

// ======================= Detection + Logger =======================
struct Detection {
    cv::Rect_<float> rect; // x,y,w,h in 640x640 space
    float prob;
    int label;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) { // reduce spam
            std::cout << msg << "\n";
        }
    }
};

// ======================= YOLODetector (TensorRT + CV-CUDA preprocess) =======================
class YOLODetector {
public:
    explicit YOLODetector(const std::string& engine_file_path);
    ~YOLODetector();

    // Returns detections in 640x640 coordinate space
    std::vector<Detection> detect(const cv::Mat& img_bgr, float conf_threshold);

private:
    void preprocessCudaToInputMem(const cv::Mat& img_bgr);

    Logger logger;
    ICudaEngine* engine = nullptr;
    IRuntime* runtime = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    std::unique_ptr<cvcuda::ResizeCropConvertReformat> preprocess_op;

    nvcv::Tensor input_tensor;   // wraps input_img
    nvcv::Tensor output_tensor;  // wraps input_mem (TRT input)

    void* input_img = nullptr;   // device buffer for original image (u8)
    void* input_mem = nullptr;   // TRT input buffer (float)
    void* output_mem = nullptr;  // TRT output buffer

    int last_input_width = 0;
    int last_input_height = 0;

    static constexpr int MODEL_W = 640;
    static constexpr int MODEL_H = 640;

    static constexpr int MAX_OUTPUT_DETECTIONS = 300;
    static constexpr int INPUT_SIZE_BYTES  = 1 * 3 * MODEL_H * MODEL_W * sizeof(float);
    static constexpr int OUTPUT_SIZE_BYTES = 1 * MAX_OUTPUT_DETECTIONS * 6 * sizeof(float);

    const char* INPUT_BLOB_NAME  = "images";
    const char* OUTPUT_BLOB_NAME = "output0";
};

YOLODetector::YOLODetector(const std::string& engine_file_path) {
    std::ifstream file(engine_file_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[ERROR] Unable to open engine: " << engine_file_path << "\n";
        std::exit(1);
    }

    file.seekg(0, file.end);
    const size_t size = (size_t)file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineModelStream(size);
    file.read(engineModelStream.data(), size);
    file.close();

    runtime = createInferRuntime(logger);
    if (!runtime) { std::cerr << "[ERROR] createInferRuntime failed\n"; std::exit(1); }

    engine = runtime->deserializeCudaEngine(engineModelStream.data(), size);
    if (!engine) { std::cerr << "[ERROR] deserializeCudaEngine failed\n"; std::exit(1); }

    context = engine->createExecutionContext();
    if (!context) { std::cerr << "[ERROR] createExecutionContext failed\n"; std::exit(1); }

    cudaMalloc(&input_mem, INPUT_SIZE_BYTES);
    cudaMalloc(&output_mem, OUTPUT_SIZE_BYTES);
    cudaStreamCreate(&stream);

    preprocess_op = std::make_unique<cvcuda::ResizeCropConvertReformat>();

    // A max image buffer (u8). Keep your limits or make dynamic if you want.
    const int MAX_INPUT_WIDTH  = 1920;
    const int MAX_INPUT_HEIGHT = 1080;
    cudaMalloc(&input_img, MAX_INPUT_WIDTH * MAX_INPUT_HEIGHT * 3);

    // Wrap TRT input_mem as CV-CUDA output tensor (NCHW float32)
    nvcv::TensorShape outShape{{1, 3, MODEL_H, MODEL_W}, NVCV_TENSOR_NCHW};
    nvcv::TensorDataStridedCuda::Buffer outBuffer;
    outBuffer.basePtr = static_cast<NVCVByte*>(input_mem);
    outBuffer.strides[0] = 3 * MODEL_H * MODEL_W * sizeof(float);
    outBuffer.strides[1] = MODEL_H * MODEL_W * sizeof(float);
    outBuffer.strides[2] = MODEL_W * sizeof(float);
    outBuffer.strides[3] = sizeof(float);

    nvcv::TensorDataStridedCuda outTensorData(
        outShape,
        nvcv::DataType{NVCV_DATA_TYPE_F32},
        outBuffer
    );
    output_tensor = nvcv::TensorWrapData(outTensorData);
}

YOLODetector::~YOLODetector() {
    if (input_mem) cudaFree(input_mem);
    if (output_mem) cudaFree(output_mem);
    if (input_img) cudaFree(input_img);
    if (stream) cudaStreamDestroy(stream);

    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
}

void YOLODetector::preprocessCudaToInputMem(const cv::Mat& img_bgr) {
    nvtx3::scoped_range r{"preprocess_gpu"};

    // Copy image to device buffer (u8 NHWC)
    cv::Mat host = img_bgr.isContinuous() ? img_bgr : img_bgr.clone();

    const int MAX_INPUT_WIDTH  = 1920;
    const int MAX_INPUT_HEIGHT = 1080;
    if (host.cols > MAX_INPUT_WIDTH || host.rows > MAX_INPUT_HEIGHT) {
        float scale = std::min(
            (float)MAX_INPUT_WIDTH / host.cols,
            (float)MAX_INPUT_HEIGHT / host.rows
        );
        cv::resize(host, host, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    cudaMemcpyAsync(input_img, host.data,
                    host.total() * host.elemSize(),
                    cudaMemcpyHostToDevice, stream);

    last_input_width  = host.cols;
    last_input_height = host.rows;

    // Wrap input_img as CV-CUDA input tensor (NHWC u8)
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

    // Fused: resize -> color convert -> normalize -> NHWC->NCHW into TRT input_mem
    (*preprocess_op)(
        stream,
        input_tensor,
        output_tensor,
        {MODEL_W, MODEL_H},
        NVCV_INTERP_LINEAR,
        {0, 0},
        NVCV_CHANNEL_REVERSE,   // BGR->RGB
        1.0f / 255.0f,
        0.0f,
        false
    );

    // No sync needed here because inference uses same stream (order is preserved).
}

std::vector<Detection> YOLODetector::detect(const cv::Mat& img_bgr, float conf_threshold) {
    preprocessCudaToInputMem(img_bgr);

    std::vector<float> output(MAX_OUTPUT_DETECTIONS * 6);

    {
        nvtx3::scoped_range r{"inference"};
        context->setTensorAddress(INPUT_BLOB_NAME, input_mem);
        context->setTensorAddress(OUTPUT_BLOB_NAME, output_mem);
        context->enqueueV3(stream);
    }

    {
        nvtx3::scoped_range r{"D2H"};
        cudaMemcpyAsync(output.data(), output_mem, OUTPUT_SIZE_BYTES, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    nvtx3::scoped_range r{"postprocess"};
    std::vector<Detection> results;
    results.reserve(64);

    for (int i = 0; i < MAX_OUTPUT_DETECTIONS; i++) {
        int off = i * 6;
        float x1 = output[off + 0];
        float y1 = output[off + 1];
        float x2 = output[off + 2];
        float y2 = output[off + 3];
        float conf = output[off + 4];
        int label = (int)output[off + 5];

        if (conf < conf_threshold) continue;

        float w = x2 - x1;
        float h = y2 - y1;
        if (w <= 1 || h <= 1) continue;

        Detection d;
        d.rect = cv::Rect_<float>(x1, y1, w, h);
        d.prob = conf;
        d.label = label;
        results.push_back(d);
    }

    return results;
}

// ======================= Dataset + Metrics =======================
static constexpr int NUM_CLASSES = 5; // unknown,yellow,blue,orange,large_orange
static constexpr int BG_CLASS = NUM_CLASSES;

const std::vector<std::string> CLASS_NAMES = {
    "unknown_cone",
    "yellow_cone",
    "blue_cone",
    "orange_cone",
    "large_orange_cone"
};

static inline std::string className(int c) {
    if (c == BG_CLASS) return "background";
    if (c >= 0 && c < (int)CLASS_NAMES.size()) return CLASS_NAMES[c];
    return "class_" + std::to_string(c);
}

float calculateIoU(const cv::Rect_<float>& a, const cv::Rect_<float>& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width,  b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float uni = areaA + areaB - inter;
    return (uni > 0) ? (inter / uni) : 0.0f;
}

// Parse YOLO label: cls xc yc w h (normalized), convert to 640x640 absolute
std::vector<Detection> parseYOLOLabels640(const std::string& label_path) {
    std::vector<Detection> gt;
    std::ifstream file(label_path);
    if (!file.is_open()) return gt;

    int cls;
    float xc, yc, w, h;
    while (file >> cls >> xc >> yc >> w >> h) {
        float abs_w = w * 640.0f;
        float abs_h = h * 640.0f;
        float abs_x = xc * 640.0f - abs_w / 2.0f;
        float abs_y = yc * 640.0f - abs_h / 2.0f;

        Detection d;
        d.rect  = cv::Rect_<float>(abs_x, abs_y, abs_w, abs_h);
        d.label = cls;
        d.prob  = 1.0f;
        gt.push_back(d);
    }
    return gt;
}

std::vector<std::string> getFilesInDir(const std::string& dir, const std::string& ext) {
    std::vector<std::string> files;
    cv::glob(dir + "/*" + ext, files, false);
    return files;
}

std::string getBaseName(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    size_t lastDot = path.find_last_of(".");
    if (lastSlash == std::string::npos) lastSlash = 0;
    else lastSlash++;
    if (lastDot == std::string::npos || lastDot < lastSlash) lastDot = path.length();
    return path.substr(lastSlash, lastDot - lastSlash);
}

struct Metrics {
    int tp=0, fp=0, fn=0, gt=0, pred=0;

    float precision() const { return (tp+fp) ? (float)tp/(tp+fp) : 0.f; }
    float recall()    const { return (tp+fn) ? (float)tp/(tp+fn) : 0.f; }
    float f1() const {
        float p=precision(), r=recall();
        return (p+r) ? 2.f*p*r/(p+r) : 0.f;
    }
};

// Confusion matrix indices: 0..NUM_CLASSES-1 plus BG_CLASS
using Confusion = std::vector<std::vector<long long>>;

// ======================= HSV Refinement (placeholder) =======================
// Reclassify unknown cones using mean HSV within bbox on a resized 640x640 image.
int hsv_classify_cone(const cv::Mat& bgr640, const cv::Rect& roi, bool use_large_orange) {
    cv::Rect r = roi & cv::Rect(0,0,bgr640.cols,bgr640.rows);
    if (r.width <= 2 || r.height <= 2) return 0; // unknown

    cv::Mat crop = bgr640(r);
    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

    cv::Scalar meanHSV = cv::mean(hsv);
    float H = (float)meanHSV[0]; // 0..179 in OpenCV
    float S = (float)meanHSV[1];
    float V = (float)meanHSV[2];

    // If too gray/dim, keep unknown
    if (S < 40 || V < 40) return 0;

    // Basic hue ranges (tune for your camera/dataset)
    // yellow ~ [20..40], orange ~ [5..20], blue ~ [90..130]
    if (H >= 20 && H <= 40) return 1;            // yellow
    if (H >= 90 && H <= 130) return 2;           // blue
    if (H >= 5  && H <  20) {
        if (use_large_orange) {
            // crude size-based separation (tune)
            float area = (float)r.area();
            if (area > 60.0f * 60.0f) return 4;  // large_orange
        }
        return 3;                                 // orange
    }

    return 0; // unknown
}

void apply_unknown_to_hsv_rule(std::vector<Detection>& preds,
                               const cv::Mat& original_bgr,
                               float tau,
                               bool enable_hsv)
{
    if (!enable_hsv) return;

    // Make a 640x640 BGR image to match your model-space boxes
    cv::Mat bgr640;
    cv::resize(original_bgr, bgr640, cv::Size(640,640), 0, 0, cv::INTER_LINEAR);

    for (auto& p : preds) {
        if (p.label != 0) continue;     // unknown_cone only
        if (p.prob < tau) continue;     // "If above Tau, move to HSV"

        cv::Rect roi((int)std::round(p.rect.x),
                     (int)std::round(p.rect.y),
                     (int)std::round(p.rect.width),
                     (int)std::round(p.rect.height));

        int new_cls = hsv_classify_cone(bgr640, roi, /*use_large_orange=*/true);
        if (new_cls != 0) {
            p.label = new_cls;
            // optionally adjust confidence; leaving as-is is fine for now
        }
    }
}

// ======================= Matching + Metrics + Confusion =======================
// Greedy assign each pred (sorted by conf) to best IoU GT (any class) >= iou_th.
// If matched:
//   - if class matches -> TP
//   - else -> FP (pred class) and FN (gt class)
// Always update confusion[gt][pred].
// If not matched -> FP (pred class), confusion[BG][pred].
// Remaining unmatched GT -> FN (gt class), confusion[gt][BG].
void match_and_accumulate(std::vector<Detection> preds,
                          const std::vector<Detection>& gts,
                          float iou_th,
                          Metrics& overall,
                          std::unordered_map<int, Metrics>& per_cls,
                          Confusion& confusion)
{
    std::sort(preds.begin(), preds.end(),
              [](const Detection& a, const Detection& b){ return a.prob > b.prob; });

    std::vector<bool> gt_used(gts.size(), false);

    overall.gt   += (int)gts.size();
    overall.pred += (int)preds.size();
    for (auto& g : gts)   per_cls[g.label].gt++;
    for (auto& p : preds) per_cls[p.label].pred++;

    // Assign preds
    for (auto& p : preds) {
        float best_iou = 0.f;
        int best_i = -1;

        for (int i = 0; i < (int)gts.size(); i++) {
            if (gt_used[i]) continue;
            float iou = calculateIoU(p.rect, gts[i].rect);
            if (iou >= iou_th && iou > best_iou) {
                best_iou = iou;
                best_i = i;
            }
        }

        if (best_i >= 0) {
            gt_used[best_i] = true;
            int gt_cls = gts[best_i].label;
            int pr_cls = p.label;

            // confusion updates (gt -> pred)
            if (gt_cls >= 0 && gt_cls <= BG_CLASS && pr_cls >= 0 && pr_cls <= BG_CLASS)
                confusion[gt_cls][pr_cls]++;

            if (gt_cls == pr_cls) {
                overall.tp++;
                per_cls[pr_cls].tp++;
            } else {
                // wrong class: FP for predicted class, FN for gt class
                overall.fp++;
                overall.fn++;
                per_cls[pr_cls].fp++;
                per_cls[gt_cls].fn++;
            }
        } else {
            // no GT match -> FP
            overall.fp++;
            per_cls[p.label].fp++;
            int pr_cls = p.label;
            if (pr_cls >= 0 && pr_cls <= BG_CLASS)
                confusion[BG_CLASS][pr_cls]++;
        }
    }

    // Remaining GT are FN
    for (int i = 0; i < (int)gts.size(); i++) {
        if (!gt_used[i]) {
            int gt_cls = gts[i].label;
            overall.fn++;
            per_cls[gt_cls].fn++;
            if (gt_cls >= 0 && gt_cls <= BG_CLASS)
                confusion[gt_cls][BG_CLASS]++;
        }
    }
}

// ======================= CLI parsing (minimal) =======================
struct Args {
    std::string engine_path;
    std::string path;          // image path (bench) or dataset_dir (val)
    bool val_mode = false;

    float conf = 0.5f;
    float iou  = 0.5f;

    bool hsv = false;
    float tau = 0.8f;

    int warmup = 5;
    int iters  = 20;
    int max_images = -1;
};

static bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; i++) if (flag == argv[i]) return true;
    return false;
}

static std::string getOpt(int argc, char** argv, const std::string& key, const std::string& def="") {
    for (int i = 1; i+1 < argc; i++) {
        if (key == argv[i]) return argv[i+1];
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

// ======================= Validation Runner =======================
void runValidation(const Args& a) {
    std::string images_dir = a.path + "/images/test";
    std::string labels_dir = a.path + "/labels/test";

    YOLODetector yolo(a.engine_path);

    std::vector<std::string> imgs = getFilesInDir(images_dir, ".jpg");
    auto pngs = getFilesInDir(images_dir, ".png");
    imgs.insert(imgs.end(), pngs.begin(), pngs.end());

    if (imgs.empty()) {
        std::cerr << "[ERROR] No images found in: " << images_dir << "\n";
        return;
    }

    Metrics overall;
    std::unordered_map<int, Metrics> per_class;
    Confusion confusion(NUM_CLASSES+1, std::vector<long long>(NUM_CLASSES+1, 0));

    int processed = 0;
    int limit = (a.max_images > 0) ? std::min((int)imgs.size(), a.max_images) : (int)imgs.size();

    std::cout << "[INFO] Validation on " << limit << " images\n"
              << "       conf=" << a.conf << " iou=" << a.iou
              << " hsv=" << (a.hsv ? "on" : "off") << " tau=" << a.tau << "\n";

    for (int idx = 0; idx < limit; idx++) {
        const auto& img_path = imgs[idx];
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;

        std::string base = getBaseName(img_path);
        std::string label_path = labels_dir + "/" + base + ".txt";

        auto gt = parseYOLOLabels640(label_path);
        auto preds = yolo.detect(img, a.conf);

        // apply rule: unknown->HSV if above tau
        apply_unknown_to_hsv_rule(preds, img, a.tau, a.hsv);

        match_and_accumulate(preds, gt, a.iou, overall, per_class, confusion);

        processed++;
        if (processed % 100 == 0) {
            std::cout << "[INFO] Processed " << processed << "/" << limit << "\n";
        }
    }

    // Print summary
    std::cout << "\n========== VALIDATION RESULTS ==========\n";
    std::cout << "Images: " << processed << "\n";
    std::cout << "GT: " << overall.gt << " Pred: " << overall.pred
              << " TP: " << overall.tp << " FP: " << overall.fp << " FN: " << overall.fn << "\n";
    std::cout << std::fixed << std::setprecision(2)
              << "Precision: " << overall.precision()*100.f << "%\n"
              << "Recall:    " << overall.recall()*100.f << "%\n"
              << "F1:        " << overall.f1()*100.f << "%\n";

    // Per-class
    std::cout << "\n------------ Per-Class Metrics ------------\n";
    for (int c = 0; c < NUM_CLASSES; c++) {
        const auto& m = per_class[c]; // default zeros if missing
        std::cout << className(c) << ":\n"
                  << "  GT " << m.gt << " Pred " << m.pred
                  << " TP " << m.tp << " FP " << m.fp << " FN " << m.fn
                  << " | P " << m.precision()*100.f << "% "
                  << "R " << m.recall()*100.f << "% "
                  << "F1 " << m.f1()*100.f << "%\n";
    }

    // Confusion matrix (includes background row/col)
    std::cout << "\n------------ Confusion Matrix (GT rows -> Pred cols) ------------\n";
    // header
    std::cout << std::setw(18) << "GT\\Pred";
    for (int c = 0; c <= NUM_CLASSES; c++) {
        std::cout << std::setw(18) << className(c);
    }
    std::cout << "\n";

    for (int r = 0; r <= NUM_CLASSES; r++) {
        std::cout << std::setw(18) << className(r);
        for (int c = 0; c <= NUM_CLASSES; c++) {
            std::cout << std::setw(18) << confusion[r][c];
        }
        std::cout << "\n";
    }

    // Unknown-focused quick summary
    const auto& mu = per_class[0];
    std::cout << "\n------------ Unknown Cone Focus ------------\n";
    std::cout << "unknown_cone P/R/F1 = "
              << mu.precision()*100.f << "% / "
              << mu.recall()*100.f << "% / "
              << mu.f1()*100.f << "%\n";
    std::cout << "Top confusions INTO unknown (GT != unknown, Pred unknown):\n";
    for (int gt_cls = 1; gt_cls < NUM_CLASSES; gt_cls++) {
        long long cnt = confusion[gt_cls][0];
        if (cnt > 0) {
            std::cout << "  " << className(gt_cls) << " -> unknown_cone: " << cnt << "\n";
        }
    }
    std::cout << "Top confusions OUT of unknown (GT unknown, Pred != unknown):\n";
    for (int pr_cls = 1; pr_cls < NUM_CLASSES; pr_cls++) {
        long long cnt = confusion[0][pr_cls];
        if (cnt > 0) {
            std::cout << "  unknown_cone -> " << className(pr_cls) << ": " << cnt << "\n";
        }
    }

    std::cout << "==========================================\n";
}

// ======================= Benchmark Runner =======================
void runBench(const Args& a) {
    YOLODetector yolo(a.engine_path);
    cv::Mat img = cv::imread(a.path);
    if (img.empty()) {
        std::cerr << "[ERROR] image empty: " << a.path << "\n";
        return;
    }

    std::cout << "[INFO] Warmup " << a.warmup << " runs...\n";
    for (int i = 0; i < a.warmup; i++) {
        auto preds = yolo.detect(img, a.conf);
        apply_unknown_to_hsv_rule(preds, img, a.tau, a.hsv);
    }

    std::cout << "[INFO] Benchmark " << a.iters << " runs...\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < a.iters; i++) {
        auto preds = yolo.detect(img, a.conf);
        apply_unknown_to_hsv_rule(preds, img, a.tau, a.hsv);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double latency = (double)ms / a.iters;
    double fps = (latency > 0) ? 1000.0 / latency : 0.0;

    std::cout << "Average latency: " << latency << " ms\n";
    std::cout << "Average FPS:     " << fps << "\n";
}

// ======================= main =======================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr <<
            "[USAGE]\n"
            "  Bench: ./inference <engine_path> <image_path> [--conf 0.5] [--hsv 0|1] [--tau 0.8] [--warmup 5] [--iters 20]\n"
            "  Val:   ./inference <engine_path> <dataset_dir> val [--conf 0.5] [--iou 0.5] [--hsv 0|1] [--tau 0.8] [--max_images N]\n"
            "        dataset_dir must contain images/test and labels/test\n";
        return 1;
    }

    Args a;
    a.engine_path = argv[1];
    a.path = argv[2];

    // mode
    a.val_mode = (argc >= 4 && std::string(argv[3]) == "val");

    // options
    a.conf = getOptFloat(argc, argv, "--conf", a.conf);
    a.iou  = getOptFloat(argc, argv, "--iou", a.iou);
    a.tau  = getOptFloat(argc, argv, "--tau", a.tau);
    a.warmup = getOptInt(argc, argv, "--warmup", a.warmup);
    a.iters  = getOptInt(argc, argv, "--iters", a.iters);
    a.max_images = getOptInt(argc, argv, "--max_images", a.max_images);

    // hsv flag: accept "--hsv 1" or "--hsv 0"
    std::string hsv_s = getOpt(argc, argv, "--hsv", "");
    if (!hsv_s.empty()) a.hsv = (hsv_s == "1" || hsv_s == "true" || hsv_s == "on");

    if (a.val_mode) runValidation(a);
    else runBench(a);

    return 0;
}