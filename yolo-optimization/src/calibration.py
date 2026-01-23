import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
from pathlib import Path
import argparse


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_files, batch_size, input_shape, cache_file="calibration.cache"):
        super(Calibrator, self).__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape  # Should be (height, width) e.g., (640, 640)
        self.cache_file = cache_file
        self.calibration_files = calibration_files
        self.current_index = 0
        
        # Calculate total size: batch * channels * height * width
        self.batch_shape = (batch_size, 3, input_shape[0], input_shape[1])
        total_size = batch_size * 3 * input_shape[0] * input_shape[1] * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(total_size)
        
        # Pre-allocate batch data with correct shape
        self.batch_data = np.zeros(self.batch_shape, dtype=np.float32)
        
        print(f"Calibrator initialized with {len(calibration_files)} images")
        print(f"Batch size: {batch_size}, Input shape: {input_shape}")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.calibration_files):
            print("Calibration complete!")
            return None

        for i in range(self.batch_size):
            img_path = self.calibration_files[self.current_index + i]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))  # (width, height)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            self.batch_data[i] = img

        # Copy to device
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(self.batch_data))
        
        batch_num = (self.current_index // self.batch_size) + 1
        total_batches = len(self.calibration_files) // self.batch_size
        print(f"Processing calibration batch {batch_num}/{total_batches}")
        
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            print(f"Using existing calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"✓ Calibration cache saved to: {self.cache_file}")


def build_engine_with_calibration(onnx_path, calibration_files, cache_file, 
                                   batch_size=8, input_shape=(640, 640), 
                                   workspace_gb=4, engine_file=None):
    """
    Build TensorRT INT8 engine with calibration (creates cache as side effect)
    
    Args:
        onnx_path: Path to ONNX model
        calibration_files: List of calibration image paths
        cache_file: Output path for calibration cache
        batch_size: Calibration batch size
        input_shape: Input image shape (H, W)
        workspace_gb: Workspace size in GB
        engine_file: Optional path to save engine file
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print(f"\n[1/4] Loading ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(f"  {parser.get_error(error)}")
            return None
    print("✓ ONNX model loaded successfully")
    
    # Create builder config
    print(f"\n[2/4] Configuring TensorRT builder")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    print(f"✓ INT8 and FP16 modes enabled, workspace: {workspace_gb}GB")
    
    # Create and set calibrator
    print(f"\n[3/4] Creating calibrator and running calibration")
    calibrator = Calibrator(
        calibration_files=calibration_files,
        batch_size=batch_size,
        input_shape=input_shape,
        cache_file=cache_file
    )
    config.int8_calibrator = calibrator
    
    # Build engine (this triggers calibration and cache creation)
    print(f"\n[4/4] Building INT8 engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None
    
    print("Engine built successfully")
    
    # Optionally save engine file
    if engine_file:
        with open(engine_file, "wb") as f:
            f.write(serialized_engine)
        print(f"✓ Engine saved to: {engine_file}")
    
    return serialized_engine


def main():
    parser = argparse.ArgumentParser(
        description='Create INT8 calibration cache for YOLO models'
    )
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--cache', type=str, default='calibration.cache')
    parser.add_argument('--engine', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workspace', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--max-images', type=int, default=500)
    
    args = parser.parse_args()
    
    # Load calibration image paths
    print(f"Loading calibration images from: {args.images}")
    img_dir = Path(args.images)
    calibration_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        calibration_files.extend([str(p) for p in sorted(img_dir.glob(ext))])
    
    if len(calibration_files) == 0:
        print(f"ERROR: No images found in {args.images}")
        return
    
    # Limit to max_images
    calibration_files = calibration_files[:args.max_images]
    print(f"Found {len(calibration_files)} calibration images")
    
    # Build engine with calibration
    build_engine_with_calibration(
        onnx_path=args.onnx,
        calibration_files=calibration_files,
        cache_file=args.cache,
        batch_size=args.batch_size,
        input_shape=(args.img_size, args.img_size),
        workspace_gb=args.workspace,
        engine_file=args.engine
    )


if __name__ == "__main__":
    main()
