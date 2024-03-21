from lib4parrot.Minidrone import Mambo  # Mambo 드론 라이브러리 import
from lib4parrot.DroneVision import DroneVision  # 드론 비전 처리 라이브러리 import
from lib4parrot.Model import Model  # 모델 import
import threading  # 스레드 처리를 위한 라이브러리 import
import cv2  # OpenCV 라이브러리 import
import numpy as np  # NumPy 라이브러리 import

import time  # 시간 관련 라이브러리 import
import sys  # 시스템 관련 라이브러리 import

# 객체 감지를 위한 YOLO 모델과 클래스 정보를 로드합니다.
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 225, size=(len(classes), 3))

# 데모를 위한 비행 여부 설정
testFlying = True

class UserVision:
    def __init__(self, vision):
        self.box_w = 100
        self.box_x = 320
        self.box_y = 180
        self.box_h = 100
        self.index = 0
        self.vision = vision
        img = self.vision.get_latest_valid_picture()

    def save_pictures(self, args):
        print("in save pictures on image %d " % self.index)

        img = self.vision.get_latest_valid_picture()
        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape

        # 이미지 전처리 후 YOLO 모델에 입력
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > 0.5:
                    # 객체 감지
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # 바운딩 박스 좌표
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    self.box_w = w
                    self.box_h = h
                    self.box_x = center_x
                    self.box_y = center_y
                    print("w = %d, h= %d ,x = %d, y = %d" % (w, h, center_x, center_y))
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                cv2.circle(img, (center_x, center_y), 5, (0, 0, 225), -1)
        if (img is not None):
            cv2.imshow("Results", img)
            if cv2.waitKey(200) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

# Mambo 드론의 주소를 설정
mamboAddr = "00:e0:2d:0b:04:6f"

# Mambo 객체 생성 및 연결
mambo = Mambo(mamboAddr, use_wifi=True)
print("trying to connect to mambo now")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

if (success):
    # 드론 상태 정보 업데이트
    print("sleeping")
    mambo.smart_sleep(1)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)

    # 비전 시작 준비
    mamboVision = DroneVision(mambo, Model.MAMBO, buffer_size=30)
    userVision = UserVision(mamboVision)
    mamboVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
    success = mamboVision.open_video()
    print("Success in opening vision is %s" % success)

    if (success):
        print("Vision successfully started!")
        if (testFlying):
            print("taking off!")
            mambo.safe_takeoff(5)

            if (mambo.sensors.flying_state != "emergency"):
                print("flying state is %s" % mambo.sensors.flying_state)
                print("Flying direct: going up")
                mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=30, duration=1)
                mambo.smart_sleep(5)
                
                # 객체가 감지될 때까지 비행 및 조정 수행
                while userVision.box_h >= 100:
                    if userVision.box_w >= 115 + 20:
                        print("Flying direct: backward---------")
                        mambo.fly_direct(roll=0, pitch=-50, yaw=0, vertical_movement=0, duration=0.5)
                        mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
                    elif userVision.box_w <= 90 - 20:
                        print("Flying direct: forward++++++++")
                        mambo.fly_direct(roll=0, pitch=50, yaw=0, vertical_movement=0, duration=0.5)
                        mambo.fly_direct(roll=0, pitch=0, yaw
                    else:
                        mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
                    
                    if userVision.box_x >= 430 + 20:
                        print("Flying direct: right>>>>>>>>")
                        mambo.fly_direct(roll=30, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
                        mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
                        mambo.fly_direct(roll=0, pitch=0, yaw=15, vertical_movement=0, duration=0.5)
                        mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
                    elif userVision.box_x <= 210 - 20:
                        print("Flying direct: left<<<<<<<<")
                        mambo.fly_direct(roll=-30, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
                        mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
                        mambo.fly_direct(roll=0, pitch=0, yaw=-15, vertical_movement=0, duration=0.5)
                        mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
                    else:
                        mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=0.5)
                    
                print("landing vvvvvvvvvvvvvv")
                print("flying state is %s" % mambo.sensors.flying_state)
                mambo.safe_land(5)
            else:
                print("Sleeeping for 15 seconds - move the mambo around")
                mambo.smart_sleep(300)

            # 비전 데모 종료
            print("Ending the sleep and vision")
            mamboVision.close_video()

            mambo.smart_sleep(5)

        print("disconnecting")
        mambo.disconnect()
