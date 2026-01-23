from ultralytics import YOLO

model =YOLO("models/player_detector.pt")
result =model.predict("input/video_1.mp4",save=True)

print(result)
print("****************************************")
for box in result[0].boxes:
    print(box)