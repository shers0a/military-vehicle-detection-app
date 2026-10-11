"""Builds the synthetic T0/T1 pair used for the Tactical Map screenshot in the README.

MVRSD has no multi-temporal pairs (the same area imaged at two dates), so T1 is an edited
copy of a real T0 tile: one group of vehicles is erased (departures) and another group is
cloned onto empty ground (arrivals). It exists only to demonstrate the change-detection
view - it is not real-world evidence of anything.

The box indices below refer to the ground-truth label file of MVRSD tile AUAU010001.

Usage (from the repo root, with the inference_service venv):
    inference_service/.venv/bin/python tools/make_synthetic_pair.py \
        dataset/val/images/AUAU010001.jpg dataset/val/labels/AUAU010001.txt t0.png t1.png
"""
import sys

import cv2
import numpy as np

DEPARTED = (12, 10, 2, 9, 6, 11)  # column of construction vehicles, top-left
ARRIVED = (13, 21, 24, 23)        # row of small vehicles, cloned onto empty tarmac
ARRIVAL_CENTER = (278, 445)
TARMAC_SHIFT_DOWN = 95            # empty tarmac lies this many px below the departed vehicles
PAD = 8


def main(image_path, labels_path, out_t0, out_t1):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    boxes = []
    for line in open(labels_path):
        _, cx, cy, bw, bh = map(float, line.split())
        boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)))

    # Departures: cover each vehicle with tarmac copied from below it, feathered at the edges.
    # (Poisson blending smeared the dark neighbouring vehicles into the fill, and inpainting
    # left a visible blur.)
    fill = img.copy()
    alpha = np.zeros((h, w), np.float32)
    for i in DEPARTED:
        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = max(x1 - PAD, 0), max(y1 - PAD, 0), min(x2 + PAD, w), min(y2 + PAD, h)
        fill[y1:y2, x1:x2] = img[y1 + TARMAC_SHIFT_DOWN:y2 + TARMAC_SHIFT_DOWN, x1:x2]
        alpha[y1:y2, x1:x2] = 1.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), 3)[..., None]
    t1 = (fill * alpha + img * (1 - alpha)).astype(np.uint8)

    # Arrivals: clone a group of vehicles onto empty ground with seamless (Poisson) blending.
    group = [boxes[i] for i in ARRIVED]
    gx1, gy1 = min(b[0] for b in group) - 6, min(b[1] for b in group) - 6
    gx2, gy2 = max(b[2] for b in group) + 6, max(b[3] for b in group) + 6
    patch = img[gy1:gy2, gx1:gx2]
    t1 = cv2.seamlessClone(patch, t1, np.full(patch.shape[:2], 255, np.uint8), ARRIVAL_CENTER, cv2.NORMAL_CLONE)

    cv2.imwrite(out_t0, img)
    cv2.imwrite(out_t1, t1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
