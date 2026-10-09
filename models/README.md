Put the trained YOLOv8 weights here as `best.pt`.

`compose.yaml` mounts this folder read-only into the inference container at `/models`.
Weights are git-ignored (`*.pt`), so they never end up in the repository or in an image.
