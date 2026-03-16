include bench/vars.mk

TRT_WS_FLAG := $(shell trtexec --help 2>/dev/null | grep -q -- '--memPoolSize' && echo "--memPoolSize=workspace:$(TRT_WS)" || echo "--workspace=$(TRT_WS)")

.PHONY: engine show build run validate bench_val profile

# ---------------- Common paths ----------------
APP_BIN ?= yolo-inference/build/inference
APP_SRC ?= yolo-inference/inference.cpp

# ---------------- Commands ----------------
TRT_CMD := trtexec --onnx=$(ONNX) --saveEngine=$(ENGINE) $(TRT_PREC) $(TRT_WS_FLAG) $(TRT_EXTRA) # $(TRT_SHAPES)

RUN_CMD := $(APP_BIN) $(ENGINE) $(IMG) --map_conf $(MAP_CONF) --warmup $(WARMUP) --iters $(ITERS)

VAL_CMD := $(APP_BIN) $(ENGINE) $(DATASET) val --map_conf $(MAP_CONF) --cm_iou $(CM_IOU) --max_images $(VAL_MAX)

# validation binary, but on one image path -> bench mode
BENCH_VAL_CMD := $(APP_BIN) $(ENGINE) $(IMG) --map_conf $(MAP_CONF) --warmup $(WARMUP) --iters $(ITERS)

NSYS_CMD := sudo nsys profile $(NSYS_OPTS) --output=$(NSYS_OUT) $(RUN_CMD)

show:
	@echo "trtexec build command:"
	@echo $(TRT_CMD)
	@echo
	@echo "build command:"
	@echo "g++ $(APP_SRC) -o $(APP_BIN) ..."
	@echo
	@echo "run command:"
	@echo $(RUN_CMD)
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

build:
	mkdir -p $(dir $(APP_BIN))
	cd $(dir $(APP_BIN)) && g++ ../inference.cpp -o inference -O3 -std=c++17 \
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
	$(RUN_CMD) | tee $(ENGINE).run.log

validate: build
	$(VAL_CMD) | tee $(ENGINE).val.log

bench_val: build
	$(BENCH_VAL_CMD) | tee $(ENGINE).bench_val.log

profile: build
	mkdir -p $(dir $(NSYS_OUT))
	$(NSYS_CMD) | tee $(NSYS_OUT).nsys.log
