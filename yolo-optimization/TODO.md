## Code Implementation
- [X] super basic dataset script to just get train script going, nothing to actually do since everything alr mounted to dale
- [X] base train script
- [ ] evl script using test data directory
- [ ] export script
- [ ] notebook/shell script to run whichever ones needed sequentially

## Execution
- [ ] **Populate Raw Data**: Ensure `ml_data/fsoco_raw` has the dataset.
- [ ] **Run Training**: `python3 src/train.py --model yolov8n.pt --params config/hyperparams.yaml`
- [ ] **Test Export**: `python3 src/export.py --weights experiments/fsoco_train/weights/best.pt --format onnx`