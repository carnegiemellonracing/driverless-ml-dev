from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plot

model = YOLO("")

areas = np.array([])
confs = np.array([])

results = model.predict(
    source="",
    stream=True,
    verbose=False
)

for r in results:
    if r.boxes == None:
        continue
    
    points = r.boxes.xywh

    area = points[2] * points[3]
    areas = np.append(areas, area)
    confs = np.append(confs, r.boxes.conf)


plt.title("Conf vs Pred Area") 
plt.xlabel("Confidence") 
plt.ylabel("Box Area") 
plt.plot(confs, areas, color ="red") 
plt.show()
