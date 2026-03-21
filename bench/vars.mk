# Paths
ONNX      := ml_data/models/yolov5_model_params.onnx
ENGINE    := ml_data/models/best26s.engine
IMG       := ml_data/fsoco_yolo/images/test/amz_amz_00004.jpg
DATASET   := ml_data/fsoco_yolo
INFER_BIN := yolo-inference/build/inference
VAL_BIN   := yolo-inference/build/validation

# Validation thresholds
MAP_CONF  := 0.001
CM_IOU   := 0.5
VAL_MAX   := -1
ALT_CONE_MAP := 0

# TensorRT build knobs (edit these)
TRT_PREC  := --fp16             # mixed precision depending on hardware support 
TRT_WS    := 2048                  # MiB
TRT_EXTRA := --verbose                   # e.g. --verbose --useCudaGraph --noTF32

# Nsight Systems knobs (edit these)
NSYS_OUT  := ml_data/model_saves/profiles/yolo26n
NSYS_OPTS := --trace=cuda,nvtx,osrt \
			 --sample=process-tree \
             --cuda-memory-usage=true \
			 --soc-metrics=true \
			 --soc-metrics-frequency=20000 \
			 --soc-metrics-set=t234 \
             --stats=true \
			 --force-overwrite=true
