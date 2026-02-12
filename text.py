from ultralytics import YOLO

# 预测
model = YOLO("D:/rzf/deeplearning2/CAF-YOLO/runs/detect/train/weights/best.pt")
source = ("D:/rzf/deeplearning2/CAF-YOLO/dataset/images/BloodImage_00002.jpg")
results = model.predict(source, save=True, show_conf=True)  # predict on an image
print('ok')