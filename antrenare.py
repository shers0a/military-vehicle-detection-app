#antrenare model code
from ultralytics import YOLO

def start_training():
    model = YOLO('yolov8m.pt')
    # antrenare
    model.train(
        data='config.yaml',
        epochs=100,
        imgsz=1280,
        batch=4,
        patience=20,
        device=0,
        name='rezultat_militar',

        #setari imagini
        degrees=45.0,
        flipud=0.5,
        mosaic=1.0,
    )
if __name__ == '__main__':
    start_training()
