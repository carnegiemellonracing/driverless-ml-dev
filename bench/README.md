- Edit knobs in `bench/vars.mk`
- Build engine: `make -f bench/Makefile engine`
- Build inference binary: `make -f bench/Makefile infer_bin`
- Build validation binary: `make -f bench/Makefile val_bin`
- Run inference benchmark: `make -f bench/Makefile run`
- Run validation (metrics on dataset): `make -f bench/Makefile validate`
- Run validation benchmark (latency on single image): `make -f bench/Makefile bench_val`
- Profile: `make -f bench/Makefile profile`
- Print commands: `make -f bench/Makefile show`

Quick overrides without editing files:
- `make -f bench/Makefile engine TRT_PREC=--fp16 TRT_WS=4096`
- `make -f bench/Makefile validate VAL_CONF=0.3 VAL_HSV=1 VAL_TAU=0.7`
- `make -f bench/Makefile profile NSYS_OUT=ml_data/model_saves/profiles/test1`