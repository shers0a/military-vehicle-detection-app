import numpy as np
import pytest

from app.tactical import calculate_density, calculate_tactical_heatmap

# 0.5 m/px and 50 m sectors -> each sector is 100x100 px.
GSD = 0.5
GRID = 50.0


def points(*xy):
    return np.array(xy, dtype=float).reshape(-1, 2)


NO_POINTS = np.empty((0, 2))


def test_grid_dimensions_follow_image_size_and_resolution():
    diff, bins_x, bins_y = calculate_tactical_heatmap(NO_POINTS, NO_POINTS, 1000, 500, GSD, GRID)

    assert (bins_x, bins_y) == (10, 5)
    assert diff.shape == (10, 5)


def test_no_vehicles_gives_all_zero_matrix():
    diff, _, _ = calculate_tactical_heatmap(NO_POINTS, NO_POINTS, 1000, 500, GSD, GRID)

    assert not diff.any()


def test_arrivals_are_positive_and_departures_negative():
    t0 = points([50, 50])                # one vehicle in sector (0, 0)
    t1 = points([50, 50], [60, 70], [950, 450])  # it stays, one more joins it, one appears in (9, 4)

    diff, _, _ = calculate_tactical_heatmap(t0, t1, 1000, 500, GSD, GRID)

    assert diff[0, 0] == 1
    assert diff[9, 4] == 1
    assert diff.sum() == 2


def test_vehicle_moving_between_sectors():
    t0 = points([150, 50])   # sector (1, 0)
    t1 = points([350, 250])  # sector (3, 2)

    diff, _, _ = calculate_tactical_heatmap(t0, t1, 1000, 500, GSD, GRID)

    assert diff[1, 0] == -1
    assert diff[3, 2] == 1
    assert diff.sum() == 0


def test_diff_is_indexed_x_then_y():
    # Regression guard for the [x][y] layout the Blazor heatmap relies on.
    t1 = points([850, 150])  # x -> sector 8, y -> sector 1

    diff, _, _ = calculate_tactical_heatmap(NO_POINTS, t1, 1000, 500, GSD, GRID)

    assert diff[8, 1] == 1


def test_image_smaller_than_one_sector_still_has_one_cell():
    diff, bins_x, bins_y = calculate_tactical_heatmap(NO_POINTS, points([10, 10]), 40, 40, GSD, GRID)

    assert (bins_x, bins_y) == (1, 1)
    assert diff[0, 0] == 1


def test_density_per_hectare():
    # 2x2 sectors of 50 m = 100 m x 100 m = 1 ha
    area_ha, density = calculate_density(10, 2, 2, GRID)

    assert area_ha == pytest.approx(1.0)
    assert density == pytest.approx(10.0)


def test_density_with_zero_area_does_not_divide_by_zero():
    area_ha, density = calculate_density(5, 2, 2, 0.0)

    assert area_ha == 0
    assert density == 0.0
