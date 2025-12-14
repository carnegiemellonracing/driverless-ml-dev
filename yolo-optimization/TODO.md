## Code Implementation
- [X] super basic dataset script to just get train script going, nothing to actually do since everything alr mounted to dale
- [X] base train script
- [X] evl script using test data directory
- [X] export script
- [X] remove explicit mlflow 'set's, just update ultralytics settings

## Execution
- [ ] **Run Training**: `python3 src/train.py --model yolov8n.pt --params config/hyperparams.yaml`
- [ ] **Test Export**: `python3 src/export.py --weights experiments/fsoco_train/weights/best.pt --format onnx`