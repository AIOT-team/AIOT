import cv2
import paho.mqtt.client as mqtt
import threading
import base64
import json
import os
import sys
from utils.trt_ssd_object_detect import TrtThread, BBoxVisualization

import time
current_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(current_path)

from utils.camera import Camera
from sensor.distance import Distance
from utils.sign_label_map import CLASSES_LIST, CLASSES_DICT
# project_path = "/home/jetson/MyWorkspace/jet3"
# sys.path.append(project_path)
###########오토-매뉴얼 모드 전환가능하게 설정 &  ***모든 subscribe 여기서 함***   ##################
class ImageMqttPublisher:
    def __init__(self, brokerIp=None, brokerPort=1883, pubTopic=None, subtopic=None, car=None,flag="stop", flag2="none"):
        self.sensor = Distance(0)
        self.brokerIp = brokerIp
        self.brokerPort = brokerPort
        self.pubTopic = pubTopic
        self.__subtopic = subtopic
        self.__client = mqtt.Client()
        self.__client.on_connect = self.__on_connect
        self.__client.on_disconnect = self.__on_disconnect
        self.__client.on_message = self.__on_message
        self.__car = car
        self.pub_data = dict()
        # self.client.on_message = self.__on_message
        self.camera = Camera(cap_w=320, cap_h=240, dp_w=320, dp_h=240, fps=30, flip_method=0)
        self.camera.camera_init()

        print("camera instance constructed")
        enginePath = current_path + "/models/ssd_mobilenet_v2_sign6/tensorrt_fp16.engine"

        self.condition = threading.Condition()
        self.trtThread = TrtThread(enginePath, TrtThread.INPUT_TYPE_USBCAM, self.camera, 0.5, self.condition, self.__car)
        # self.__cnt = 0
        #자동모드시 a키 d키(좌 우 회전) 안먹히는 플래그
        self.auto_flag = True
        self.__flag = flag
        self.__flag2 = flag2

    def start(self):
        thread = threading.Thread(target=self.__run)
        thread.start()

    def __run(self):
        self.__client.connect(self.brokerIp, self.brokerPort)

        # TrtThread 객체 생성
        trtThread = self.trtThread

        # 감지 시작
        trtThread.start()
        full_scrn = False

        # 초당 프레임 수 확인
        fps = 0.0

        # 시작 시간
        tic = time.time()
        vis = BBoxVisualization(CLASSES_DICT)

        self.__client.loop_start()

        while trtThread.running:
            # dist = self.sensor.read()
            # print("거리 : ", dist, "cm")
            with self.condition:
                # 감지 결과가 있을 때 까지 대기
                self.condition.wait()
                # 감지 결과 얻기
                img, boxes, confs, clss,\
                    class_name, line_left, line_right = trtThread.getDetectResult()

            img = vis.drawBox(img, boxes, clss)

            # 초당 프레임 수 드로잉
            # img = vis.drawFps(img, fps)
            # print(fps)
            # 이미지를 윈도우에 보여주기
            # cv2.imshow("detect_from_video", img)

            # 초당 프레임 수 계산
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
            tic = toc

            #self.sendBase64(img)
            retvv2, bytes2 = cv2.imencode(".jpg", img)
            b64_bytes2 = base64.b64encode(bytes2).decode("utf-8")
            self.pub_data["Cam"] = b64_bytes2
            self.pub_data.update({"battery": self.__car.get_voltage_percentage()})
            self.pub_data.update({"servo": self.__car.get_servo_angle()})
            self.pub_data.update({"speed": self.__car.get_dc_speed()})
            self.pub_data.update({"label": class_name})
            self.pub_data.update({"boxes": boxes})
            self.pub_data.update({"line_left": line_left})
            self.pub_data.update({"line_right": line_right})

            data = json.dumps(self.pub_data)
            self.__client.publish(self.pubTopic, data)

            # 키보드 입력을 위해 1ms 동안 대기, 입력이 없으면 -1을 리턴
            key = cv2.waitKey(1)
            if key == 27:
                # esc(27)를 눌렀을 때
                break
            elif key == ord("F") or key == ord("f"):  # ord: 주어진 키의 아스키 키코드를 리턴해준다
                # F나 f를 눌렀을 경우 전체 스크린 토글 기능
                full_scrn = not full_scrn  # 현재값이 true면 false로 바꾸고 false면 true로 바꾸기
                if full_scrn:
                    cv2.setWindowProperty("detect_from_video", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)  # (해당윈도우의 이름, 풀스크린 시키는 기능)
                else:
                    cv2.setWindowProperty("detect_from_video", cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL)  # 노말이면 원래 크기로 돌아옴

    def __on_connect(self, client, userdata, flags, rc):
        print("ImageMqttClient mqtt broker connect")
        # self.client.subscribe("command/camera/capture")
        self.__client.subscribe(self.__subtopic)

    def __on_disconnect(self, client, userdata, rc):
        print("ImageMqttClient mqtt broker disconnect")

    # def __on_message(self, client, userdata, message):
    #     if "capture" in message.topic:
    #         retval, frame = self.camera.videoCapture.read()
    #         if retval:
    #             img = np.copy(frame)
    #             cv2.imwrite("/home/pi/Project/SensingRover/capture/capture_image" + str(self.__cnt) + ".jpg", img)
    #             self.__cnt += 1
    #             capval, bytes = cv2.imencode(".jpg", frame)
    #             if capval:
    #                 cap_b64_bytes = base64.b64encode(bytes)
    #                 self.client.publish("/capturepub", cap_b64_bytes)
    #                 print("pub complete")

    def __on_message(self, client, userdata, message):
        print("뭔가 옴")
        if "go" in message.topic:
            self.__flag = "go"
            print("가자2")

        elif "stop" in message.topic:
            self.__flag = "stop"
            print("멈추자2")

        elif "toL" in message.topic:
            self.__flag2 = "left"
            print("좌측으로 차선변경")

        elif "toR" in message.topic:
            self.__flag2 = "right"
            print("우측으로 차선변경")
        # elif "D" in message.topic:



    def get_flag(self):
        return self.__flag

    def get_flag2(self):
        return self.__flag2

    def set_flag2(self, flag2):
        self.__flag2 = flag2

    def disconnect(self):
        self.__client.disconnect()

    def sendBase64(self, frame):
        if self.__client is None:
            return

        # MQTT Broker가 연결되어 있지 않을 경우
        if not self.__client.is_connected():
            return

        # JPEG 포맷으로 인코딩
        retval, bytes = cv2.imencode(".jpg", frame)

        # 인코딩이 실패했을 경우
        if not retval:
            print("image encoding fail")
            return

        # Base64 문자열로 인코딩
        b64_bytes = base64.b64encode(bytes)

        # MQTT Broker로 보내기
        self.__client.publish(self.pubTopic, b64_bytes)

    def stop(self):
        self.__client.unsubscribe(self.__subtopic)
        self.__client.disconnect()






if __name__ == '__main__':
    videoCapture = cv2.VideoCapture(0)
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    imageMqttPublisher = ImageMqttPublisher("192.168.3.183", 1883, "/camerapub")
    imageMqttPublisher.connect()

    while True:
        if videoCapture.isOpened():
            retval, frame = videoCapture.read()
            if not retval:
                print("video capture fail")
                break
            imageMqttPublisher.sendBase64(frame)
        else:
            print("videoCapture is not opened")
            break

    imageMqttPublisher.disconnect()
    videoCapture.release()

