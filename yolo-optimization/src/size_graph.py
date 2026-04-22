import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

source_dir = "ml_data/fsoco_yolo/images/test"
#progress calc
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
total_images = len(image_files)

model = YOLO("ml_data/model_saves/models/26s_tuned_best.pt")

areas = np.array([])
confs = np.array([])

print(f"Starting inference on {total_images} images...", flush=True)

results = model.predict(
    source=source_dir,
    stream=True,
    verbose=False
)

for i, r in enumerate(results):
    #logging
    if (i + 1) % max(1, total_images // 10) == 0:
        percent = 100 * (i + 1) // total_images
        print(f"Progress: {percent}% ({i + 1}/{total_images})", flush=True)

    if r.boxes is None or len(r.boxes) == 0:
        continue
    
    # boxes.xywh returns [x, y, w, h]
    boxes = r.boxes.xywh
    area = boxes[:, 2] * boxes[:, 3]
    
    areas = np.append(areas, area.cpu().numpy())
    confs = np.append(confs, r.boxes.conf.cpu().numpy())


from scipy.stats import binned_statistic

plt.figure(figsize=(12, 7))

# plot data (x: area, y: conf)
# log scale: 10 to 700,000+
plt.scatter(areas, confs, color="red", alpha=0.2, s=2, label='Detections') 

# calc binned mean to show the trend
# log-spaced bins for better resolution at smaller area sizes
bins = np.logspace(np.log10(max(1, areas.min())), np.log10(areas.max()), 30)
bin_means, bin_edges, _ = binned_statistic(areas, confs, statistic='mean', bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.plot(bin_centers, bin_means, color='blue', lw=3, label='Trend (Mean Confidence)')

# cutoff calc
TARGET_CONF = 0.8
valid_bins = bin_means >= TARGET_CONF
recommended_cutoff = 0

if np.any(valid_bins):
    # smallest area (first bin) that meets the conf requirement
    cutoff_idx = np.where(valid_bins)[0][0]
    recommended_cutoff = bin_edges[cutoff_idx]
    
    plt.axvline(x=recommended_cutoff, color='green', linestyle='--', lw=2, 
                label=f'Suggested Cutoff: {recommended_cutoff:.0f}px')
    
    plt.text(recommended_cutoff * 1.1, 0.05, f'Cutoff: {recommended_cutoff:.0f}px', 
             color='green', fontweight='bold', rotation=90, verticalalignment='bottom')
    
    plt.fill_betweenx([0, 1], 0, recommended_cutoff, color='gray', alpha=0.1, label='Unreliable Zone')

plt.xscale('log')
plt.title(f"26s test - Box Area vs Confidence (Conf: {TARGET_CONF})") 
plt.xlabel("Box Area (Pixels) - Log Scale") 
plt.ylabel("Confidence Score") 
plt.ylim(0, 1.05)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.legend()

plt.savefig(os.path.join("ml_data/model_saves", "test_area_vs_conf_26s.png"))
print(f"\nGraph updated: area_vs_conf.png")
if recommended_cutoff > 0:
    print(f"RECOMMENDED CUTOFF: {recommended_cutoff:.0f} pixels")
    print(f"Removing detections smaller than this will filter out most low-confidence noise.")
else:
    print("\nCould not determine a safe cutoff - confidence never reached the target threshold.")
