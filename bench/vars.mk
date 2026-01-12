# Paths
ONNX      := ml_data/model_saves/11v_runs/yolo11n/weights/model.onnx
ENGINE    := ml_data/model_saves/engines/yolo11n.fp16.engine
IMG       := ml_data/fsoco_yolo/images/test/amz_amz_00097.jpg
INFER_BIN := yolo-inference/build/inference

# TensorRT build knobs (edit these)
TRT_PREC  := --fp16                # or empty / --int8
TRT_WS    := 2048                  # MiB
TRT_SHAPES:= --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640
TRT_EXTRA :=                         # e.g. --verbose --useCudaGraph --noTF32

# Nsight Systems knobs (edit these)
NSYS_OUT  := ml_data/model_saves/profiles/yolo11n
NSYS_OPTS := --trace=cuda,nvtx,osrt \
			 --accelerator-trace=tegra-accelerators \	# may have no effect
			 --sample=process-tree \
             --cuda-memory-usage=true \
			 --soc-metrics=true \
			 --soc-metrics-frequency=20000 \
			 --soc-metrics-set=t234 \
             --stats=true \
			 --force-overwrite=true
