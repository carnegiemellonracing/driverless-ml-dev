#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <cmath>

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
    float prob = 0.0f;
    int label = -1;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << "\n";
        }
    }
};

// ======================= YOLODetector =======================
class YOLODetector {
public:
    explicit YOLODetector(const std::string& engine_file_path);
    ~YOLODetector();

    std::vector<Detection> detect(const cv::Mat& img_bgr, float conf_threshold);

private:
    void preprocessCudaToInputMem(const cv::Mat& img_bgr);

    Logger logger;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;

    std::unique_ptr<cvcuda::ResizeCropConvertReformat> preprocess_op;

    nvcv::Tensor input_tensor;
    nvcv::Tensor output_tensor;

    void* input_img = nullptr;
    void* input_mem = nullptr;
    void* output_mem = nullptr;

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
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, file.beg);

    std::vector<char> engineModelStream(size);
    file.read(engineModelStream.data(), size);
    file.close();

    runtime = createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "[ERROR] createInferRuntime failed\n";
        std::exit(1);
    }

    engine = runtime->deserializeCudaEngine(engineModelStream.data(), size);
    if (!engine) {
        std::cerr << "[ERROR] deserializeCudaEngine failed\n";
        std::exit(1);
    }

    context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "[ERROR] createExecutionContext failed\n";
        std::exit(1);
    }

    cudaMalloc(&input_mem, INPUT_SIZE_BYTES);
    cudaMalloc(&output_mem, OUTPUT_SIZE_BYTES);
    cudaStreamCreate(&stream);

    preprocess_op = std::make_unique<cvcuda::ResizeCropConvertReformat>();

    const int MAX_INPUT_WIDTH  = 1920;
    const int MAX_INPUT_HEIGHT = 1080;
    cudaMalloc(&input_img, MAX_INPUT_WIDTH * MAX_INPUT_HEIGHT * 3);

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

    cudaMemcpyAsync(input_img, host.data, host.total() * host.elemSize(),
                    cudaMemcpyHostToDevice, stream);

    last_input_width  = host.cols;
    last_input_height = host.rows;

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

    (*preprocess_op)(
        stream,
        input_tensor,
        output_tensor,
        {MODEL_W, MODEL_H},
        NVCV_INTERP_LINEAR,
        {0, 0},
        NVCV_CHANNEL_REVERSE,  // BGR -> RGB
        1.0f / 255.0f,
        0.0f,
        false
    );
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

    std::vector<Detection> results;
    results.reserve(64); // TODO: validate this line

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

        Detection d;
        d.rect  = cv::Rect_<float>(x1, y1, w, h);
        d.prob  = conf;
        d.label = label;
        results.push_back(d);
    }

    return results;
}

// ======================= Dataset helpers =======================
static constexpr int NUM_CLASSES = 5;
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
    if (c >= 0 && c < static_cast<int>(CLASS_NAMES.size())) return CLASS_NAMES[c];
    return "class_" + std::to_string(c);
}

//look through the algo to see if it's accurate
float calculateIoU(const cv::Rect_<float>& a, const cv::Rect_<float>& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width,  b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float uni   = areaA + areaB - inter;
    return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

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
    size_t lastDot   = path.find_last_of(".");
    if (lastSlash == std::string::npos) lastSlash = 0;
    else lastSlash++;
    if (lastDot == std::string::npos || lastDot < lastSlash) lastDot = path.length();
    return path.substr(lastSlash, lastDot - lastSlash);
}

// ======================= HSV refinement =======================
int hsv_classify_cone(const cv::Mat& bgr640, const cv::Rect& roi, bool use_large_orange) {
    cv::Rect r = roi & cv::Rect(0, 0, bgr640.cols, bgr640.rows);
    if (r.width <= 2 || r.height <= 2) return 0;

    cv::Mat crop = bgr640(r);
    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

    cv::Scalar meanHSV = cv::mean(hsv);
    float H = static_cast<float>(meanHSV[0]);
    float S = static_cast<float>(meanHSV[1]);
    float V = static_cast<float>(meanHSV[2]);

    if (S < 40 || V < 40) return 0;
    if (H >= 20 && H <= 40) return 1;   // yellow
    if (H >= 90 && H <= 130) return 2;  // blue
    if (H >= 5 && H < 20) {
        if (use_large_orange && static_cast<float>(r.area()) > 60.0f * 60.0f) return 4;
        return 3; // orange
    }
    return 0;
}

void apply_unknown_to_hsv_rule(std::vector<Detection>& preds,
                               const cv::Mat& original_bgr,
                               float tau,
                               bool enable_hsv)
{
    if (!enable_hsv) return;

    cv::Mat bgr640;
    cv::resize(original_bgr, bgr640, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);

    for (auto& p : preds) {
        if (p.label != 0) continue;
        if (p.prob < tau) continue;

        cv::Rect roi(static_cast<int>(std::round(p.rect.x)),
                     static_cast<int>(std::round(p.rect.y)),
                     static_cast<int>(std::round(p.rect.width)),
                     static_cast<int>(std::round(p.rect.height)));

        int new_cls = hsv_classify_cone(bgr640, roi, true);
        if (new_cls != 0) p.label = new_cls;
    }
}

// ======================= mAP =======================
struct PredRecord {
    int image_id = -1;
    int cls = -1;
    float score = 0.0f;
    cv::Rect_<float> box;
};

struct GTRecord {
    int image_id = -1;
    int cls = -1;
    cv::Rect_<float> box;
};

struct APResult {
    double ap = 0.0;
    int num_gt = 0;
    int num_pred = 0;
};

//look through the algo to see if it's accurate
APResult computeAPForClass(const std::vector<PredRecord>& all_preds,
                           const std::vector<GTRecord>& all_gts,
                           int target_cls,
                           float iou_thresh)
{
    std::vector<PredRecord> preds;
    std::vector<GTRecord> gts;

    for (const auto& p : all_preds) {
        if (p.cls == target_cls) preds.push_back(p);
    }
    for (const auto& g : all_gts) {
        if (g.cls == target_cls) gts.push_back(g);
    }

    APResult result;
    result.num_gt = static_cast<int>(gts.size());
    result.num_pred = static_cast<int>(preds.size());

    if (gts.empty()) return result;

    std::sort(preds.begin(), preds.end(),
              [](const PredRecord& a, const PredRecord& b) {
                  return a.score > b.score;
              });

    std::unordered_map<int, std::vector<cv::Rect_<float>>> gt_boxes_by_image;
    std::unordered_map<int, std::vector<bool>> gt_used_by_image;

    for (const auto& g : gts) {
        gt_boxes_by_image[g.image_id].push_back(g.box);
    }
    for (auto& kv : gt_boxes_by_image) {
        gt_used_by_image[kv.first] = std::vector<bool>(kv.second.size(), false);
    }

    std::vector<double> tp(preds.size(), 0.0);
    std::vector<double> fp(preds.size(), 0.0);

    for (size_t i = 0; i < preds.size(); i++) {
        const auto& p = preds[i];

        auto it = gt_boxes_by_image.find(p.image_id);
        if (it == gt_boxes_by_image.end()) {
            fp[i] = 1.0;
            continue;
        }

        auto& gt_boxes = it->second;
        auto& gt_used  = gt_used_by_image[p.image_id];

        double best_iou = 0.0;
        int best_j = -1;

        for (size_t j = 0; j < gt_boxes.size(); j++) {
            if (gt_used[j]) continue;
            double iou = calculateIoU(p.box, gt_boxes[j]);
            if (iou > best_iou) {
                best_iou = iou;
                best_j = static_cast<int>(j);
            }
        }

        if (best_j >= 0 && best_iou >= iou_thresh) {
            gt_used[best_j] = true;
            tp[i] = 1.0;
        } else {
            fp[i] = 1.0;
        }
    }

    for (size_t i = 1; i < tp.size(); i++) {
        tp[i] += tp[i - 1];
        fp[i] += fp[i - 1];
    }

    std::vector<double> recalls(tp.size(), 0.0);
    std::vector<double> precisions(tp.size(), 0.0);

    for (size_t i = 0; i < tp.size(); i++) {
        recalls[i] = tp[i] / std::max(1, result.num_gt);
        precisions[i] = tp[i] / std::max(1.0, tp[i] + fp[i]);
    }

    std::vector<double> mrec, mpre;
    mrec.reserve(recalls.size() + 2);
    mpre.reserve(precisions.size() + 2);

    mrec.push_back(0.0);
    mpre.push_back(0.0);
    for (size_t i = 0; i < recalls.size(); i++) {
        mrec.push_back(recalls[i]);
        mpre.push_back(precisions[i]);
    }
    mrec.push_back(1.0);
    mpre.push_back(0.0);

    for (int i = static_cast<int>(mpre.size()) - 2; i >= 0; i--) {
        mpre[i] = std::max(mpre[i], mpre[i + 1]);
    }

    double ap = 0.0;
    for (size_t i = 1; i < mrec.size(); i++) {
        if (mrec[i] != mrec[i - 1]) {
            ap += (mrec[i] - mrec[i - 1]) * mpre[i];
        }
    }

    result.ap = ap;
    return result;
}

void reportMAP50(const std::vector<PredRecord>& all_preds,
                 const std::vector<GTRecord>& all_gts)
{
    double sum_ap = 0.0;
    int valid_classes = 0;

    std::cout << "\n------------ AP@0.50 ------------\n";
    for (int c = 0; c < NUM_CLASSES; c++) {
        APResult r = computeAPForClass(all_preds, all_gts, c, 0.50f);
        if (r.num_gt > 0) {
            std::cout << std::setw(20) << std::left << className(c)
                      << " AP50=" << std::fixed << std::setprecision(4) << r.ap
                      << "  GT=" << r.num_gt
                      << "  Pred=" << r.num_pred << "\n";
            sum_ap += r.ap;
            valid_classes++;
        } else {
            std::cout << std::setw(20) << std::left << className(c)
                      << " AP50=N/A (no GT)\n";
        }
    }

    double map50 = (valid_classes > 0) ? (sum_ap / valid_classes) : 0.0;
    std::cout << "mAP@0.50 = " << std::fixed << std::setprecision(4) << map50 << "\n";
}

void reportMAP5095(const std::vector<PredRecord>& all_preds,
                   const std::vector<GTRecord>& all_gts)
{
    std::vector<float> ious;
    for (int k = 0; k < 10; k++) ious.push_back(0.50f + 0.05f * k);

    double total = 0.0;
    int count = 0;

    std::cout << "\n------------ AP@[0.50:0.95] ------------\n";
    for (int c = 0; c < NUM_CLASSES; c++) {
        int gt_count = 0;
        for (const auto& g : all_gts) {
            if (g.cls == c) gt_count++;
        }

        if (gt_count == 0) {
            std::cout << std::setw(20) << std::left << className(c)
                      << " AP@[.50:.95]=N/A (no GT)\n";
            continue;
        }

        double cls_sum = 0.0;
        for (float iou : ious) {
            APResult r = computeAPForClass(all_preds, all_gts, c, iou);
            cls_sum += r.ap;
        }

        double cls_map = cls_sum / ious.size();
        std::cout << std::setw(20) << std::left << className(c)
                  << " AP@[.50:.95]=" << std::fixed << std::setprecision(4) << cls_map << "\n";

        total += cls_map;
        count++;
    }

    double map5095 = (count > 0) ? (total / count) : 0.0;
    std::cout << "mAP@0.50:0.95 = " << std::fixed << std::setprecision(4) << map5095 << "\n";
}

// ======================= Confusion Matrix =======================
using Confusion = std::vector<std::vector<long long>>;

// Build confusion matrix at one IoU threshold.
// Rows = GT, Cols = Pred, with extra background row/col.
void accumulateConfusionForImage(std::vector<Detection> preds,
                                 const std::vector<Detection>& gts,
                                 float iou_th,
                                 Confusion& confusion)
{
    std::sort(preds.begin(), preds.end(),
              [](const Detection& a, const Detection& b) {
                  return a.prob > b.prob;
              });

    std::vector<bool> gt_used(gts.size(), false);

    for (const auto& p : preds) {
        float best_iou = 0.0f;
        int best_i = -1;

        for (int i = 0; i < static_cast<int>(gts.size()); i++) {
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

            if (gt_cls >= 0 && gt_cls <= BG_CLASS &&
                pr_cls >= 0 && pr_cls <= BG_CLASS) {
                confusion[gt_cls][pr_cls]++;
            }
        } else {
            int pr_cls = p.label;
            if (pr_cls >= 0 && pr_cls <= BG_CLASS) {
                confusion[BG_CLASS][pr_cls]++;
            }
        }
    }

    for (int i = 0; i < static_cast<int>(gts.size()); i++) {
        if (!gt_used[i]) {
            int gt_cls = gts[i].label;
            if (gt_cls >= 0 && gt_cls <= BG_CLASS) {
                confusion[gt_cls][BG_CLASS]++;
            }
        }
    }
}

void printConfusionMatrix(const Confusion& confusion) {
    std::cout << "\n------------ Confusion Matrix (GT rows -> Pred cols) ------------\n";
    std::cout << std::setw(20) << "GT\\Pred";
    for (int c = 0; c <= NUM_CLASSES; c++) {
        std::cout << std::setw(20) << className(c);
    }
    std::cout << "\n";

    for (int r = 0; r <= NUM_CLASSES; r++) {
        std::cout << std::setw(20) << className(r);
        for (int c = 0; c <= NUM_CLASSES; c++) {
            std::cout << std::setw(20) << confusion[r][c];
        }
        std::cout << "\n";
    }
}

// ======================= CLI =======================
struct Args {
    std::string engine_path;
    std::string path;  // image_path for bench, dataset_dir for val
    bool val_mode = false;

    float map_conf = 0.001f;
    float cm_iou = 0.50f;

    bool hsv = false;
    float tau = 0.8f;

    int warmup = 5;
    int iters = 20;
    int max_images = -1;
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

// ======================= Validation =======================
void runValidation(const Args& a) {
    std::string images_dir = a.path + "/images/test";
    std::string labels_dir = a.path + "/labels/test";

    YOLODetector yolo(a.engine_path);

    std::vector<std::string> imgs = getFilesInDir(images_dir, ".jpg");
    auto pngs = getFilesInDir(images_dir, ".png");
    imgs.insert(imgs.end(), pngs.begin(), pngs.end());
    std::sort(imgs.begin(), imgs.end());

    if (imgs.empty()) {
        std::cerr << "[ERROR] No images found in: " << images_dir << "\n";
        return;
    }

    int limit = (a.max_images > 0)
        ? std::min(static_cast<int>(imgs.size()), a.max_images)
        : static_cast<int>(imgs.size());

    std::vector<PredRecord> all_preds;
    std::vector<GTRecord> all_gts;
    Confusion confusion(NUM_CLASSES + 1, std::vector<long long>(NUM_CLASSES + 1, 0));

    std::cout << "[INFO] Validation on " << limit << " images\n"
              << "       map_conf=" << a.map_conf
              << " cm_iou=" << a.cm_iou
              << " hsv=" << (a.hsv ? "on" : "off")
              << " tau=" << a.tau << "\n";

    int processed = 0;
    for (int idx = 0; idx < limit; idx++) {
        const auto& img_path = imgs[idx];
        cv::Mat img = cv::imread(img_path);
        if (img.empty()) continue;

        std::string base = getBaseName(img_path);
        std::string label_path = labels_dir + "/" + base + ".txt";

        auto gts = parseYOLOLabels640(label_path);
        auto preds = yolo.detect(img, a.map_conf);
        apply_unknown_to_hsv_rule(preds, img, a.tau, a.hsv);

        for (const auto& g : gts) {
            if (g.label >= 0 && g.label < NUM_CLASSES) {
                all_gts.push_back({idx, g.label, g.rect});
            }
        }

        for (const auto& p : preds) {
            if (p.label >= 0 && p.label < NUM_CLASSES) {
                all_preds.push_back({idx, p.label, p.prob, p.rect});
            }
        }

        accumulateConfusionForImage(preds, gts, a.cm_iou, confusion);

        processed++;
        if (processed % 100 == 0) {
            std::cout << "[INFO] Processed " << processed << "/" << limit << "\n";
        }
    }

    std::cout << "\n========== VALIDATION RESULTS ==========\n";
    std::cout << "Images: " << processed << "\n";
    std::cout << "GT boxes: " << all_gts.size() << "\n";
    std::cout << "Pred boxes kept: " << all_preds.size() << "\n";

    reportMAP50(all_preds, all_gts);
    reportMAP5095(all_preds, all_gts);
    printConfusionMatrix(confusion);

    std::cout << "========================================\n";
}

// ======================= Benchmark =======================
void runBench(const Args& a) {
    YOLODetector yolo(a.engine_path);
    cv::Mat img = cv::imread(a.path);
    if (img.empty()) {
        std::cerr << "[ERROR] image empty: " << a.path << "\n";
        return;
    }

    std::cout << "[INFO] Warmup " << a.warmup << " runs...\n";
    for (int i = 0; i < a.warmup; i++) {
        auto preds = yolo.detect(img, a.map_conf);
        apply_unknown_to_hsv_rule(preds, img, a.tau, a.hsv);
    }

    std::cout << "[INFO] Benchmark " << a.iters << " runs...\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < a.iters; i++) {
        auto preds = yolo.detect(img, a.map_conf);
        apply_unknown_to_hsv_rule(preds, img, a.tau, a.hsv);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double latency = static_cast<double>(ms) / a.iters;
    double fps = (latency > 0.0) ? 1000.0 / latency : 0.0;

    std::cout << "Average latency: " << latency << " ms\n";
    std::cout << "Average FPS:     " << fps << "\n";
}

// ======================= main =======================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr
            << "[USAGE]\n"
            << "  Bench: ./inference <engine_path> <image_path> "
            << "[--map_conf 0.001] [--hsv 0|1] [--tau 0.8] [--warmup 5] [--iters 20]\n"
            << "  Val:   ./inference <engine_path> <dataset_dir> val "
            << "[--map_conf 0.001] [--cm_iou 0.50] [--hsv 0|1] [--tau 0.8] [--max_images N]\n"
            << "        dataset_dir must contain images/test and labels/test\n";
        return 1;
    }

    Args a;
    a.engine_path = argv[1];
    a.path = argv[2];
    a.val_mode = (argc >= 4 && std::string(argv[3]) == "val");

    a.map_conf   = getOptFloat(argc, argv, "--map_conf", a.map_conf);
    a.cm_iou     = getOptFloat(argc, argv, "--cm_iou", a.cm_iou);
    a.tau        = getOptFloat(argc, argv, "--tau", a.tau);
    a.warmup     = getOptInt(argc, argv, "--warmup", a.warmup);
    a.iters      = getOptInt(argc, argv, "--iters", a.iters);
    a.max_images = getOptInt(argc, argv, "--max_images", a.max_images);

    std::string hsv_s = getOpt(argc, argv, "--hsv", "");
    if (!hsv_s.empty()) a.hsv = (hsv_s == "1" || hsv_s == "true" || hsv_s == "on");

    if (a.val_mode) runValidation(a);
    else runBench(a);

    return 0;
}