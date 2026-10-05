import numpy as np


def calculate_tactical_heatmap(
    points_t0: np.ndarray,
    points_t1: np.ndarray,
    img_w: int,
    img_h: int,
    gsd_m_per_px: float,
    grid_size_m: float,
):
    pixelgrid = grid_size_m / gsd_m_per_px

    bins_x = max(int(img_w / pixelgrid), 1)
    bins_y = max(int(img_h / pixelgrid), 1)

    p0_x = points_t0[:, 0] if points_t0.size > 0 else []
    p0_y = points_t0[:, 1] if points_t0.size > 0 else []

    p1_x = points_t1[:, 0] if points_t1.size > 0 else []
    p1_y = points_t1[:, 1] if points_t1.size > 0 else []

    heatmap_t0, _, _ = np.histogram2d(
        p0_x, p0_y, bins=[bins_x, bins_y], range=[[0, img_w], [0, img_h]]
    )
    heatmap_t1, _, _ = np.histogram2d(
        p1_x, p1_y, bins=[bins_x, bins_y], range=[[0, img_w], [0, img_h]]
    )

    diff_matrix = heatmap_t1 - heatmap_t0
    return diff_matrix, bins_x, bins_y


def calculate_density(vehicle_count: int, bins_x: int, bins_y: int, grid_size_m: float):
    total_area_sqm = bins_x * bins_y * (grid_size_m * grid_size_m)
    total_area_ha = total_area_sqm / 10000
    density = vehicle_count / total_area_ha if total_area_ha > 0 else 0.0
    return total_area_ha, density
