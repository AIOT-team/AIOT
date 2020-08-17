import cv2

def gstreamer_pipeline(
    capture_width=400,
    capture_height=300,
    display_width=400,
    display_height=300,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def zzcapture():

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    return cap


##
if __name__ == "__main__":
    cap = zzcapture()
    count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #cv2.imshow('', frame)

        key = cv2.waitKey(200)
        if key == 27:
            break
        cv2.imwrite("/home/jetson/capture/"+str(count)+".jpg", frame)
        count +=1
        if count>300:
            break
    cv2.destroyAllWindows()
    cap.release()
