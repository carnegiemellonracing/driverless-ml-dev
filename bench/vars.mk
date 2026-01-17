# Paths
ONNX      := ml_data/model_saves/26v_runs/yolo26n/best_26n.onnx
ENGINE    := ml_data/model_saves/engines/yolo26n.fp16.engine
IMG       := ml_data/fsoco_yolo/images/test/amz_amz_00097.jpg
INFER_BIN := yolo-inference/build/inference

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
