- Edit knobs in `bench/vars.mk`
- Build engine: `make -f bench/Makefile engine`
- Run inference: `make -f bench/Makefile run`
- Profile: `make -f bench/Makefile profile`
- Print commands: `make -f bench/Makefile show`
- Validate: `make -f bench/Makefile validate`

Quick overrides without editing files:
- `make -f bench/Makefile engine TRT_PREC=--fp16 TRT_WS=4096`
- `make -f bench/Makefile profile NSYS_OUT=ml_data/model_saves/profiles/test1`
`idk whatim doing why am i a brick burh`