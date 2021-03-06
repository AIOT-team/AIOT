import cv2
import numpy as np


road_center_x = 0

def line_detect(frame):
    # =======================편한 사이즈로 재조정=========================
    # frame = cv2.resize(frame, dsize=(320, 480))

    # =======================흑백 컬러로 변환============================
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # =========================가우시안 블러 처리, 노이즈 제거=========================
    blur_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # =========================threshold 처리=====================================
    th, img_th = cv2.threshold(blur_gray, 190, 255, cv2.THRESH_BINARY)

    # =========================close morphology : 구멍이 채워지고 좀 더 뚜렷해짐=========================
    kernel = np.ones((5, 5), np.uint8)
    img_morph = cv2.morphologyEx(img_th.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # =========================Canny 윤곽선 검출=========================
    edges = cv2.Canny(img_morph, 50, 200)

    # =========================ROI=============================================================
    mask = np.zeros_like(img_gray)
    ignore_mask_color = 255
    height, width = img_gray.shape
    # 320(w) x 240(h)일 때의 ROI/ 테스트 완료
    vertices = np.array([[(0, height), (40, height / 2), (width-40, height / 2),
                          (width, height)]], dtype=np.int32)
    # vertices = np.array([[(0, height), (60, height*0.3), (width-60, height*0.3),
    #                       (width, height)]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_img = cv2.bitwise_and(edges, mask)


   #####temp#####
    # mask2 = np.zeros_like(frame)
    # cv2.fillPoly(mask2, vertices, (255,255,255))
    # hi = cv2.bitwise_and(frame,mask2)
    ####temp#####

    # =========================Hough Transform을 이용한 직선 검출, 리턴된 lines는 (n, 1, 4)의 shape을 가진다.(n : 검출된 직선의 개수) =========================
    # threshold : 높을 수록 정확도는 올라가고, 적은 선을 찾음, 낮으면 많은 직선을 찾지만 대부분의 직선을 찾음
    # minLineLength : 찾을 직선의 최소 길이, maxLineGap : 선과의 최대 간격
    lines = cv2.HoughLinesP(masked_img, 1, np.pi / 180, 25, minLineLength=15, maxLineGap=20)

    if lines is None:
        line_retval = False
        return line_retval, None, None

    if lines.shape[0] == 1:
        line_arr = lines[0, :, :]
    else:
        line_arr = lines.squeeze()

    # 기울기 구하기
    slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # 수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]

    # 수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]

    # 필터링된 직선 버리기
    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]

    line_retval = True

    return line_retval, L_lines, R_lines


def offset_detect(img, L_lines, R_lines, road_half_width_list):
    global road_center_x
    h, w, _ = img.shape

    # 고정 y 값
    y_fix = int(h * (2 / 3))

    # 화면 중앙 점
    center_x = int(w / 2)
    center_point = (center_x, y_fix)

    # 교점들을 저장할 리스트
    left_cross_points = []
    right_cross_points = []
    right_line_x = None
    left_line_x = None

    L_lines_detected = False
    R_lines_detected = False
    if L_lines is not None:
        L_lines_detected = bool(len(L_lines) != 0)
    # 왼/오 선을 찾았는지 bool 변수에 저장
    if R_lines is not None:
        R_lines_detected = bool(len(R_lines) != 0)

    # 둘다 찾았을 경우
    if L_lines_detected and R_lines_detected:
        for each_line in L_lines:
            x1, y1, x2, y2 = each_line
            # 직선 그리기
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 직선의 기울기
            slope = (y2 - y1) / (x2 - x1)
            # 교점의 x 좌표
            cross_x = ((y_fix - y1) / slope) + x1

            # 교점의 x 좌표 저장
            left_cross_points.append(cross_x)

        for each_line in R_lines:
            x1, y1, x2, y2 = each_line
            # 직선 그리기
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 직선의 기울기
            slope = (y2 - y1) / (x2 - x1)
            # 교점의 x 좌표
            cross_x = ((y_fix - y1) / slope) + x1

            # 교점의 x 좌표 저장
            right_cross_points.append(cross_x)

        # 모든 선들의 가장 작은 x 좌표가 왼쪽, 큰 x 좌표가 오른쪽
        left_line_x = min(left_cross_points)
        right_line_x = max(right_cross_points)

        # 도로 너비의 반 계산 후 저장
        road_half_width = (right_line_x - left_line_x) / 2
        road_half_width_list.append(road_half_width)

        # 도로 중간 지점 저장
        road_center_x = left_line_x + road_half_width
        #print("둘다 찾았을 때 중앙", road_center_x)
        road_center_point = (int(road_center_x), y_fix)

    # 둘중 하나만 찾았을 경우
    elif L_lines_detected ^ R_lines_detected:
        road_half_width = np.mean(road_half_width_list)

        # 왼쪽 선만 찾았을 경우
        if L_lines_detected:
            for each_line in L_lines:
                x1, y1, x2, y2 = each_line
                # 직선 그리기
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 직선의 기울기
                slope = (y2 - y1) / (x2 - x1)
                # 교점의 x 좌표
                cross_x = ((y_fix - y1) / slope) + x1

                # 교점의 x 좌표 저장
                left_cross_points.append(cross_x)

            # 왼쪽선들만 찾았으니, 그중 가장 작은 x 좌표가 왼쪽 선
            left_line_x = min(left_cross_points)
            # 도로 중간 지점 저장
            road_center_x = left_line_x + road_half_width
            #print("왼쪽만 찾았을 때 중앙", road_center_x)

            road_center_point = (int(road_center_x), y_fix)

        # 오른쪽 선만 찾았을 경우
        else:
            for each_line in R_lines:
                x1, y1, x2, y2 = each_line
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # 직선의 기울기
                slope = (y2 - y1) / (x2 - x1)
                # 교점의 x 좌표
                cross_x = ((y_fix - y1) / slope) + x1

                # 교점의 x 좌표 저장
                right_cross_points.append(cross_x)

            # 오른쪽선들만 찾았으니, 그중 가장 큰 x 좌표가 오른쪽 선
            right_line_x = max(right_cross_points)
            # 도로 중간 지점 저장
            road_center_x = right_line_x - road_half_width
            #print("오른쪽만 찾았을 때 중앙", road_center_x)

            road_center_point = (int(road_center_x), y_fix)

    #둘다 못 찾았을 경우
    else:
        print("l,r라인 둘다 [] 꼴")
        #print("둘다 못찾으면 직전 값?", road_center_x)
        road_half_width = np.mean(road_half_width_list)
        road_center_point = (int(road_center_x), y_fix)

    # 도로 중간 지점 / 자동차 중간 지점과 라인 시각화
    cv2.circle(img, road_center_point, 5, (255, 0, 0), -1)
    cv2.circle(img, center_point, 5, (0, 255, 0), -1)
    cv2.line(img, road_center_point, center_point, (255, 255, 255), 2)

    # 왼쪽을 돌려야하면 음수, 오른쪽으로 돌려야하면 양수
    offset_width = road_center_x - center_x
    offset_height = h - y_fix

    # 각도 구하기
    # 오른쪽으로 회전해야 하는 경우 각도가 음수, 왼쪽으로 회전해야하는 경우 양수
    angle = np.arctan2(offset_height, offset_width) * 180 / (np.pi) - 90

    return angle, left_line_x, right_line_x

# %% test
if __name__ == '__main__':
    videoCapture = cv2.VideoCapture("C:/MyWorkspace/final/project_4_advanced_lane_finding/test_1.avi")
    # videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while videoCapture.isOpened():
        retval, frame = videoCapture.read()
        if not retval:
            break
        height, width = frame.shape
        lines = line_detect(frame)

        # =========================선을 찾지 못했다면, 다음 프레임으로 continue=========================
        if lines is None:
            cv2.imshow("video", frame)
            continue

        # =========================선을 하나만 찾는경우, squeeze()에 의해 선의 개수 축(0축)까지 벗겨질 수 있기 때문에 1개의 선만 찾았을 때 분할 처리
        if lines.shape[1] == 1:
            line_arr = lines[0, :, :]
        else:
            line_arr = lines.squeeze()


        # result = line_limited_angle(line_arr, frame)

        cv2.imshow("video", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    videoCapture.release()
    cv2.destroyAllWindows()

