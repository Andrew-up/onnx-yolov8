# from cap_from_youtube import cap_from_youtube
import os.path
import time
from datetime import datetime

import cv2
import yaml

from definitions import ROOT_DIR, OUTPUT
from yolov8 import YOLOv8


def test_cam():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
            break

        try:
            # Read frame from the video
            ret, frame = cap.read()
            cv2.imshow("Detected Objects", frame)
            if not ret:
                break
        except Exception as e:
            print(e)
            continue


def main():
    print('start programm')
    file_config = 'config.yaml'
    input_video_path = os.path.join(ROOT_DIR, 'data/video.mp4')
    number_web_cam = 0
    using_web_cam = False
    save_video = False
    if os.path.exists(os.path.join(ROOT_DIR, file_config)):
        with open(os.path.join(ROOT_DIR, file_config), mode='r', encoding='utf-8') as file_yaml:
            # rwerwer = yaml.safe_load(file_yaml)
            # pass
            listyaml = yaml.load(file_yaml, Loader=yaml.FullLoader)
            using_web_cam = listyaml.get('USING_WEB_CAM')
            number_web_cam = listyaml.get('NUM_WEBCAM')
            save_video = listyaml.get('SAVE_VIDEO')
            input_video_path = os.path.join(ROOT_DIR, listyaml.get('VIDEO_PATH'))

    print(f'use cam : {using_web_cam}')
    if not os.path.exists(OUTPUT):
        os.mkdir(OUTPUT)

    now = datetime.now()
    dt_string = now.strftime("%d.%m.%Y %H:%M:%S")

    file_out = os.path.join(OUTPUT, 'out.txt')
    if not os.path.isfile(os.path.join(ROOT_DIR, file_out)):
        open(file_out, "w+")

    if using_web_cam:
        # testDevice(number_web_cam)
        cap = cv2.VideoCapture(number_web_cam)
        with open(file_out, 'a', encoding='utf-8') as f:
            f.write(dt_string + f" webcam_num: {number_web_cam}" + "\n")

    else:
        cap = cv2.VideoCapture(input_video_path)
        with open(file_out, 'a', encoding='utf-8') as f:
            f.write(dt_string + f" {input_video_path}" + "\n")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print(frame_width, frame_height)

    size = (frame_width, frame_height)

    result = cv2.VideoWriter('out/filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    # Initialize YOLOv7 model
    model_path = "models/best.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.65)
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

    prev_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    if frame_width >= 1024 and frame_height >= 1024:
        cv2.resizeWindow('Detected Objects', frame_width // 2, frame_height // 2)

    print('#' * 30)

    if (frame_width <= 500) or (frame_height <= 500):
        print('frame_width*2')
        print('frame_height*2')
        cv2.resizeWindow('Detected Objects', frame_width + 300, frame_height + 300)

    while cap.isOpened():
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        # Press key q to stop
        if cv2.waitKey(1) == ord('q'):
            break

        try:
            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                break
        except Exception as e:
            print(e)
            continue

        # Update object localizer
        boxes, scores, class_ids = yolov8_detector(frame)

        combined_img = yolov8_detector.draw_detections(frame, file_save_txt=file_out)
        cv2.putText(combined_img, f'fps: {fps}', (30, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Detected Objects", combined_img)
        # if save_video:
        # result.write(combined_img)

    if save_video:
        result.release()
    cv2.destroyAllWindows()
    print('end')


if __name__ == '__main__':
    main()
