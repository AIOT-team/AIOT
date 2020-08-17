# from mqtt.MqttClient import MqttClient
from mqtt.CameraMqttClient2_auto_manual import ImageMqttPublisher
from car.car_control import Car
from utils.trt_ssd_object_detect import TrtThread
import time
car = Car()

# mqttClient = MqttClient("192.168.3.30", 1883, "/Control/#", "/sensor", car)
# mqttClient.start()
# print("MqttClient start")
cameraClient = ImageMqttPublisher("192.168.3.184", 1883, '/3jetracer', [("/3manual/#", 0),("/3jr", 0)], car)
# cameraClient =ImageMqttPublisher("192.168.3.105", 1883, '/2jetracer', [("/2manual/#", 0), ("/2jr", 0)], car)
cameraClient.start()
print("CameraMqttClient start")
trt = cameraClient.trtThread
while True:
    flag = cameraClient.get_flag()
    flag2 = cameraClient.get_flag2()
    trt.set_flag(flag)
    trt.set_flag2(flag2)
    time.sleep(0.1)
# 캡쳐 (cv2.imshow하기 때문에 camera mqtt와 동시에 실행 불가)
# capture_thread = vc.Capture_thread()
# capture_thread.camara_init()
# capture_thread.start()
