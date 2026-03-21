include bench/vars.mk

TRT_WS_FLAG := $(shell trtexec --help 2>/dev/null | grep -q -- '--memPoolSize' && echo "--memPoolSize=workspace:$(TRT_WS)" || echo "--workspace=$(TRT_WS)")

BACKEND ?= trt

.PHONY: engine show build val_bin run validate bench_val profile

# ---------------- Common paths ----------------
APP_BIN ?= yolo-inference/build/inference
APP_SRC ?= yolo-inference/inference.cpp

VAL_BIN ?= yolo-inference/build/validation
VAL_SRC ?= yolo-inference/validation.cpp

# ---------------- Backend selection ----------------
ifeq ($(BACKEND),trt)
MODEL ?= $(ENGINE)
BACKEND_DEFINE := -DUSE_TENSORRT
BACKEND_INC := \
  -I/usr/local/cuda/include \
  -I/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nvtx/include \
  -I/opt/nvidia/cvcuda0/include
BACKEND_LIB := \
  -L/usr/local/cuda/lib64 -lnvinfer -lcudart \
  -L/opt/nvidia/cvcuda0/lib -lcvcuda -lnvcv_types \
  -Wl,-rpath,/opt/nvidia/cvcuda0/lib
else ifeq ($(BACKEND),onnx)
MODEL ?= $(ONNX)
BACKEND_DEFINE := -DUSE_ONNX
BACKEND_INC :=
BACKEND_LIB :=
else
$(error BACKEND must be either 'trt' or 'onnx')
endif

# ---------------- Commands ----------------
TRT_CMD := trtexec --onnx=$(ONNX) --saveEngine=$(ENGINE) $(TRT_PREC) $(TRT_WS_FLAG) $(TRT_EXTRA) # $(TRT_SHAPES)

RUN_CMD := $(APP_BIN) $(MODEL) $(IMG) --map_conf $(MAP_CONF) --warmup $(WARMUP) --iters $(ITERS)

ALT_OPTS := $(if $(filter 1,$(ALT_CONE_MAP)),--alt_cone_map,)

VAL_CMD := $(VAL_BIN) $(MODEL) $(DATASET) val --map_conf $(MAP_CONF) --cm_iou $(CM_IOU) --max_images $(VAL_MAX) $(ALT_OPTS)

# validation binary, but on one image path -> bench mode
BENCH_VAL_CMD := $(VAL_BIN) $(MODEL) $(IMG) --map_conf $(MAP_CONF) --warmup $(WARMUP) --iters $(ITERS) $(ALT_OPTS)

NSYS_CMD := sudo nsys profile $(NSYS_OPTS) --output=$(NSYS_OUT) $(RUN_CMD)

show:
	@echo "BACKEND:"
	@echo $(BACKEND)
	@echo
	@echo "MODEL:"
	@echo $(MODEL)
	@echo
	@echo "trtexec build command:"
	@echo $(TRT_CMD)
	@echo
	@echo "build command:"
	@echo "g++ $(APP_SRC) -o $(APP_BIN) ..."
	@echo
	@echo "validation build command:"
	@echo "g++ $(VAL_SRC) -o $(VAL_BIN) ..."
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
ifeq ($(BACKEND),trt)
	mkdir -p $(dir $(ENGINE))
	$(TRT_CMD) | tee $(ENGINE).build.log
else
	@echo "engine target only applies to BACKEND=trt"
	@exit 1
endif

build:
	mkdir -p $(dir $(APP_BIN))
	cd $(dir $(APP_BIN)) && g++ ../inference.cpp -o inference -O3 -std=c++17 \
	  $(BACKEND_DEFINE) \
	  `pkg-config --cflags --libs opencv4` \
	  $(BACKEND_INC) \
	  $(BACKEND_LIB) \
	  -pedantic-errors -Wall -Wextra

val_bin:
	mkdir -p $(dir $(VAL_BIN))
	cd $(dir $(VAL_BIN)) && g++ ../validation.cpp -o validation -O3 -std=c++17 \
	  $(BACKEND_DEFINE) \
	  `pkg-config --cflags --libs opencv4` \
	  $(BACKEND_INC) \
	  $(BACKEND_LIB) \
	  -pedantic-errors -Wall -Wextra

run: build
	$(RUN_CMD) | tee $(MODEL).run.log

validate: val_bin
	$(VAL_CMD) | tee $(MODEL).val.log

bench_val: val_bin
	$(BENCH_VAL_CMD) | tee $(MODEL).bench_val.log

profile: build
	mkdir -p $(dir $(NSYS_OUT))
	$(NSYS_CMD) | tee $(NSYS_OUT).nsys.log