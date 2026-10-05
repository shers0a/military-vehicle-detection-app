"""Dev-only sanity script: runs the tactical-map pipeline locally against two
image files, without going through the HTTP API. Replaces the old hmp.py.

Usage: python examples/hmp_cli_example.py picture_1.jpg picture_2.jpg
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import detection, tactical
from app.config import get_settings


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <image_t0> <image_t1>")
        sys.exit(1)

    settings = get_settings()
    model = detection.load_model(settings)

    image_t0 = Image.open(sys.argv[1]).convert("RGB")
    image_t1 = Image.open(sys.argv[2]).convert("RGB")

    print("Analysing T0...")
    result_t0 = detection.run_sliced_detection(
        np.array(image_t0), model, settings.default_slice_size, settings.default_overlap_ratio
    )
    print("Analysing T1...")
    result_t1 = detection.run_sliced_detection(
        np.array(image_t1), model, settings.default_slice_size, settings.default_overlap_ratio
    )

    predictions_t0 = detection.filter_by_confidence(result_t0, settings.default_confidence_threshold)
    predictions_t1 = detection.filter_by_confidence(result_t1, settings.default_confidence_threshold)

    points_t0 = detection.extract_vehicle_points(predictions_t0)
    points_t1 = detection.extract_vehicle_points(predictions_t1)

    img_w, img_h = image_t0.size
    diff_matrix, bins_x, bins_y = tactical.calculate_tactical_heatmap(
        points_t0, points_t1, img_w, img_h, settings.default_gsd_m_per_px, settings.default_grid_size_m
    )
    total_area_ha, density = tactical.calculate_density(
        len(points_t1), bins_x, bins_y, settings.default_grid_size_m
    )

    print(f"Tactical Grid: {bins_x}x{bins_y} sectors.")
    print(f"Vehicles T1: {len(points_t1)}, Area: {total_area_ha:.2f} ha, Density: {density:.2f} veh/ha")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed (see requirements-dev.txt); skipping plot.")
        print(diff_matrix)
        return

    plt.figure(figsize=(12, 10))
    sns.heatmap(diff_matrix.T, cmap="vlag", center=0, annot=True, fmt=".0f")
    plt.title("Tactical Map: Vehicle Movement (Delta T1 - T0)")
    plt.xlabel("X COORDINATE (Sectors)")
    plt.ylabel("Y COORDINATE (Sectors)")
    plt.show()


if __name__ == "__main__":
    main()
