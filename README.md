# Driverless ML Development – YOLO Optimization on Jetson

This repository contains computer vision model development and optimization work for the
**Carnegie Mellon Racing Driverless** team.

The focus of this project is **deploying fast, accurate YOLO-based object detection on NVIDIA Jetson hardware**, with an emphasis on reproducible experiments, TensorRT optimization, and end-to-end latency analysis.

---

## Problem Statement

The existing camera detection pipeline relied on a largely off-the-shelf YOLOv5 model, resulting in **slow inference times** on our embedded hardware.

Our goal is to:
- Upgrade to newer YOLO architectures
- Drastically **reduce inference latency**
- Maintain or improve detection accuracy
- Minimize model size and runtime memory usage
- Validate all changes through **systematic experiments and profiling**

---

## Project Scope and Goals

1. **Model selection**
   - Evaluate newer YOLO architectures (v8, v10, v11)
   - Compare size, accuracy, and runtime characteristics
   - Determine feasibility of custom architectures if needed

2. **Model optimization**
   - TensorRT conversion and engine tuning
   - Precision reduction (FP16, INT8)
   - Layer and tensor fusion
   - NMS optimization strategies

3. **Deployment realism**
   - Test on target hardware (Jetson AGX Orin)
   - Measure true end-to-end latency (preprocess → inference → postprocess)
   - Profile GPU, memory bandwidth, and CPU overhead

4. **Reproducibility**
   - Clean Python scripts (no notebook-only workflows)
   - Consistent containerized environment
   - Repeatable benchmarks and visualizations

---

## Repository Structure

- **yolo-inference/**  
  Inference pipelines and runtime evaluation on Jetson.

- **yolo-optimization/**  
  TensorRT export, profiling, and performance optimization workflows.

- **bench/**
  Standardized Makefile tools for building, running, and profiling tensorRT engines
---

## Executive Summary (Fall Semester)

We transitioned from notebook-based experimentation and CLI workflows to **clean, script-based pipelines using the modern Ultralytics API**, significantly improving clarity and reproducibility.  
Models were trained and evaluated across multiple YOLO versions before being deployed and optimized directly on a **Jetson AGX Orin**.

We focused heavily on **TensorRT optimization**, as this dominated real-world performance, using NVIDIA tooling and profiling to guide decisions.  
As of December 16, optimized models achieve **sub-8 ms end-to-end inference** for some configurations, though further work is required to recover accuracy through hyperparameter tuning and larger model variants.

Key lesson learned: **optimize for simplicity and real hardware early**.

---

## Key Results (Jetson AGX Orin)

| Model | Params | Input | Batch | Precision | Warm-Start Latency | Notes |
|------|--------|-------|-------|-----------|--------------------|-------|
| YOLOv5 | — | 640×640 | 1 | FP32 | ~20 ms | Current baseline |
| YOLOv8n | ~3.2M | 640×640 | 1 | FP16 | ~10–12 ms | TensorRT optimized |
| YOLOv10n | ~2.7M | 640×640 | 1 | FP16 | **~7–8 ms** | Fastest variant |
| YOLOv11n | ~3.0M | 640×640 | 1 | FP16 | ~10–12 ms | TensorRT optimized |

**System Utilization (Peak / Mean):**
- **GPU (GR3D_FREQ):** 75% / 45%  
- **Memory Bandwidth (EMC_FREQ):** 9%

---

## Implementation Overview

### Training & Experimentation
- Ultralytics CLI (YOLOv5)
- Ultralytics Python API (YOLOv8 / v10 / v11)
- Clean Python training scripts
- Planned experiment tracking with MLflow

### Optimization & Deployment
- NVIDIA TensorRT SDK
- ONNX → TensorRT engine builds
- FP16 and INT8 quantization
- Nsight Systems & Nsight Compute profiling
- CPU vs GPU preprocessing analysis

---

## Open Questions & Next Steps

- Hyperparameter tuning to recover accuracy
- Evaluation of larger YOLO variants
- EfficientNMS_TRT integration (removing data-dependent ops)
- CUDA-based preprocessing and pipelining (cvCUDA)
- C++ TensorRT inference for Point-to-Pixel integration
- Batch-2 inference for stereo camera setup

## Branches
YOLO inference optimization (Jetson, TensorRT, Nsight profiling) is maintained on the `yolo-optimization-api-jetson-nsight` branch.  
YOLO training, experimentation, and hyperparameter tuning code is on `yolo-optimization-api`.
