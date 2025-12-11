import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

MODEL_PATH = './runs/detect/rezultat_militar/weights/best.pt'

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=0.35,
    device="cpu"
)


def coordinates(image_path, model):
    result = get_sliced_prediction(
        image_path,
        model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    coord = []
    for obj in result.object_prediction_list:
        if obj.category.name != "Civilian_Vehicle":
            center_x = (obj.bbox.min_x + obj.bbox.max_x) / 2
            center_y = (obj.bbox.min_y + obj.bbox.max_y) / 2
            coord.append([center_x, center_y])

    if len(coord) == 0:
        return np.empty((0, 2))

    return np.array(coord)


img_path_t0 = "picture_1.jpg"
img_path_t1 = "picture_2.jpg"

print("Analysing T0...")
points_t0 = coordinates(img_path_t0, detection_model)

print("Analysing T1...")
points_t1 = coordinates(img_path_t1, detection_model)

image = Image.open(img_path_t0)
img_w, img_h = image.size

GSD = 0.3
grid_size = 50

pixelgrid = grid_size / GSD

bins_x = int(img_w / pixelgrid)
bins_y = int(img_h / pixelgrid)

print(f"Tactical Grid: {bins_x}x{bins_y} sectors.")

heatmap_t0, _, _ = np.histogram2d(
    points_t0[:, 0], points_t0[:, 1],
    bins=[bins_x, bins_y],
    range=[[0, img_w], [0, img_h]]
)

heatmap_t1, _, _ = np.histogram2d(
    points_t1[:, 0], points_t1[:, 1],
    bins=[bins_x, bins_y],
    range=[[0, img_w], [0, img_h]]
)

diff_matrix = heatmap_t1 - heatmap_t0

plt.figure(figsize=(12, 10))
sns.heatmap(
    diff_matrix.T,
    cmap="vlag",
    center=0,
    annot=True,
    fmt='.0f'
)

plt.title("Tactical Map: Vehicle Movement (Delta T1 - T0)")
plt.xlabel("X COORDINATE (Sectors)")
plt.ylabel("Y COORDINATE (Sectors)")
plt.show()