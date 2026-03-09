include bench/vars.mk

TRT_WS_FLAG := $(shell trtexec --help 2>/dev/null | grep -q -- '--memPoolSize' && echo "--memPoolSize=workspace:$(TRT_WS)" || echo "--workspace=$(TRT_WS)")

.PHONY: engine run profile show infer_bin val_bin validate bench_val

# ---- Define the commands once ----
TRT_CMD    := trtexec --onnx=$(ONNX) --saveEngine=$(ENGINE) $(TRT_PREC) $(TRT_WS_FLAG) $(TRT_EXTRA) # $(TRT_SHAPES)
INFER_CMD  := $(INFER_BIN) $(ENGINE) $(IMG)
VAL_CMD    := $(VAL_BIN) $(ENGINE) $(DATASET) val --conf $(VAL_CONF) --iou $(VAL_IOU) --hsv $(VAL_HSV) --tau $(VAL_TAU) --max_images $(VAL_MAX)
BENCH_VAL_CMD := $(VAL_BIN) $(ENGINE) $(IMG) --conf $(VAL_CONF) --hsv $(VAL_HSV) --tau $(VAL_TAU)
NSYS_CMD   := sudo nsys profile $(NSYS_OPTS) --output=$(NSYS_OUT) $(INFER_CMD)

show:
	@echo "trtexec build command:"
	@echo $(TRT_CMD)
	@echo
	@echo "infer command:"
	@echo $(INFER_CMD)
	@echo
	@echo "validate command:"
	@echo $(VAL_CMD)
	@echo
	@echo "bench_val command:"
	@echo $(BENCH_VAL_CMD)
	@echo
	@echo "profile command:"
	@echo $(NSYS_CMD)

engine:
	mkdir -p $(dir $(ENGINE))
	$(TRT_CMD) | tee $(ENGINE).build.log

infer_bin:
	mkdir -p yolo-inference/build
	cd yolo-inference/build && g++ ../inference.cpp -o inference -O3 -std=c++17 \
	  `pkg-config --cflags --libs opencv4` \
	  -I/usr/local/cuda/include \
	  -I/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nvtx/include \
	  -I/opt/nvidia/cvcuda0/include \
	  -L/usr/local/cuda/lib64 -lnvinfer -lcudart \
	  -L/opt/nvidia/cvcuda0/lib -lcvcuda -lnvcv_types \
	  -Wl,-rpath,/opt/nvidia/cvcuda0/lib \
	  -pedantic-errors -Wall -Wextra

val_bin:
	mkdir -p yolo-inference/build
	cd yolo-inference/build && g++ ../validation.cpp -o validation -O3 -std=c++17 \
	  `pkg-config --cflags --libs opencv4` \
	  -I/usr/local/cuda/include \
	  -I/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nvtx/include \
	  -I/opt/nvidia/cvcuda0/include \
	  -L/usr/local/cuda/lib64 -lnvinfer -lcudart \
	  -L/opt/nvidia/cvcuda0/lib -lcvcuda -lnvcv_types \
	  -Wl,-rpath,/opt/nvidia/cvcuda0/lib \
	  -pedantic-errors -Wall -Wextra

run:
	$(INFER_CMD) | tee $(ENGINE).run.log

profile:
	mkdir -p $(dir $(NSYS_OUT))
	$(NSYS_CMD) | tee $(NSYS_OUT).nsys.log

validate:
	$(VAL_CMD) | tee $(ENGINE).val.log

bench_val:
	$(BENCH_VAL_CMD) | tee $(ENGINE).bench_val.log
