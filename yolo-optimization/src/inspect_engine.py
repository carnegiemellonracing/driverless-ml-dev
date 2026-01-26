import argparse
import tensorrt as trt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())

    if engine is None:
        raise SystemExit("Failed to deserialize engine")

    print(f"TensorRT python: {trt.__version__}")
    print(f"Engine name: {engine.name}")
    print(f"I/O tensors: {engine.num_io_tensors}")
    print(f"Optimization profiles: {engine.num_optimization_profiles}\n")

    # List I/O tensors
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)  # <-- key fix
        mode = engine.get_tensor_mode(name)  # INPUT / OUTPUT
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)  # may include -1 if dynamic
        fmt = engine.get_tensor_format_desc(name)
        loc = engine.get_tensor_location(name)
        print(f"[{i}] {mode}  name={name}")
        print(f"     dtype={dtype}  shape={tuple(shape)}  format={fmt}  location={loc}")

    # Show profile shapes (for inputs) if dynamic
    if engine.num_optimization_profiles > 0:
        print("\n=== Profile shapes (inputs) ===")
        for p in range(engine.num_optimization_profiles):
            print(f"\nProfile {p}:")
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                    continue
                mn, opt, mx = engine.get_tensor_profile_shape(name, p)
                print(f"  {name}: min={tuple(mn)} opt={tuple(opt)} max={tuple(mx)}")

if __name__ == "__main__":
    main()
