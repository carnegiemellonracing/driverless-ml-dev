from driverless-ml-dev:
cd yolo-inference/
g++ ../inference.cpp -o inference -O3 `pkg-config --cflags --libs opencv4` -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvinfer -lcudart -pedantic-errors -Wall -Wextra

from yolo-inference:
cd ..
usage as specified in inference.cpp with relative paths

for tegrastats:
sudo tegrastats --interval 100 --logfile util.t
xt | grep --line-buffered -oP '(EMC_FREQ|GR3D_FREQ) [^ ]+'