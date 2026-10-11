import math

import numpy as np


def calculate_tactical_heatmap(
    points_t0: np.ndarray,
    points_t1: np.ndarray,
    img_w: int,
    img_h: int,
    gsd_m_per_px: float,
    grid_size_m: float,
):
    sector_px = grid_size_m / gsd_m_per_px

    # Every sector is exactly grid_size_m wide, so the last row/column may extend past the
    # image edge. (Truncating with int() and stretching the bins over the whole image made
    # sectors silently larger than requested, e.g. 64 m instead of 50 m on a 640 px tile.)
    # The small epsilon keeps an exact fit (e.g. 1000 px / 100 px) from adding an empty column.
    bins_x = max(math.ceil(img_w / sector_px - 1e-9), 1)
    bins_y = max(math.ceil(img_h / sector_px - 1e-9), 1)
    grid_range = [[0, bins_x * sector_px], [0, bins_y * sector_px]]

    p0_x = points_t0[:, 0] if points_t0.size > 0 else []
    p0_y = points_t0[:, 1] if points_t0.size > 0 else []

    p1_x = points_t1[:, 0] if points_t1.size > 0 else []
    p1_y = points_t1[:, 1] if points_t1.size > 0 else []

    heatmap_t0, _, _ = np.histogram2d(p0_x, p0_y, bins=[bins_x, bins_y], range=grid_range)
    heatmap_t1, _, _ = np.histogram2d(p1_x, p1_y, bins=[bins_x, bins_y], range=grid_range)

    diff_matrix = heatmap_t1 - heatmap_t0
    return diff_matrix, bins_x, bins_y


def calculate_density(vehicle_count: int, img_w: int, img_h: int, gsd_m_per_px: float):
    # Area of the ground actually covered by the image, not of the sector grid
    # (edge sectors can extend past the image).
    total_area_sqm = (img_w * gsd_m_per_px) * (img_h * gsd_m_per_px)
    total_area_ha = total_area_sqm / 10000
    density = vehicle_count / total_area_ha if total_area_ha > 0 else 0.0
    return total_area_ha, density
