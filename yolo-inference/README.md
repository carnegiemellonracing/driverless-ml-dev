from driverless-ml-dev:
cd yolo-inference/build/
g++ ../inference.cpp -o inference -O3 `pkg-config --cflags --libs opencv4` -I/usr/local/cuda/include -I/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nvtx/include -L/usr/local/cuda/lib64 -lnvinfer -lcudart -pedantic-errors -Wall -Wextra

from yolo-inference:
cd ..
usage as specified in inference.cpp with relative paths

to make nsight analysis file:
sudo nsys profile \
    --accelerator-trace='tegra-accelerators' \
    --trace=cuda,nvtx,osrt \
    --sample=process-tree \
    --cuda-memory-usage=true \
    --soc-metrics=true \
    --soc-metrics-frequency=20000 \
    --soc-metrics-set=t234 \
    --stats=true \
    --force-overwrite=true \
    --output=v11n-3 \
    ./yolo-inference/build/inference ml_data/model_saves/11v_runs/yolo11n/weights/model_nms.engine ml_data/fsoco_yolo/images/test/amz_amz_00097.jpg
