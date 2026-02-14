from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import glob

model = YOLO("yolo11n.pt")

results = model.val(data="your_coco.yaml", save_json=True)

pred_json = glob.glob(str(results.save_dir / "predictions.json"))[0]  # created by save_json=True
gt_json = "path/to/instances_val.json"  # your COCO GT annotations

coco_gt = COCO(gt_json)
coco_dt = coco_gt.loadRes(pred_json)

e = COCOeval(coco_gt, coco_dt, iouType="bbox")
e.evaluate()
e.accumulate()
e.summarize()

map_s, map_m, map_l = e.stats[3], e.stats[4], e.stats[5]
print("mAP_s, mAP_m, mAP_l:", map_s, map_m, map_l)