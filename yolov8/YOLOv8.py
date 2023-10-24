import os
import time
from functools import lru_cache

import cv2
import numpy as np

os.environ['OMP_NUM_THREADS'] = '2'
import onnxruntime

from yolov8.utils import xywh2xyxy, draw_detection, multiclass_nms, xyxy2xywh, class_names, draw_navigation
import platform


class Boxes:

    def __init__(self, data, orig_shape) -> None:
        self.data = data
        self.orig_shape = orig_shape

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xyxy(self):
        """Return the boxes in xyxy format."""
        return self.data[:, :4]

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """Return the boxes in xywh format."""
        """Я вообще не уверен что работает правильно"""
        return xyxy2xywh(self.xyxy)


class ProcessDetect:

    def __init__(self):
        self.object = None
        self.boxes = None
        self.center = None
        self.color = None
        self.name = None
        self.score = None
        self.img_size = None


class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.providers = [
            ('TensorrtExecutionProvider', {
                'trt_engine_cache_enable': True,
            }),
            'CUDAExecutionProvider'
        ]

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def get_center_object(self):
        return self.center_detect

    def get_boxes_xyxy(self):
        return self.boxes_xyxy

    def get_score(self):
        return self.score

    def get_class_id(self):
        return self.class_id

    def get_class_name(self):
        if self.class_id is not None:
            return class_names[self.class_id]
        return None

    def get_distance_xy(self):
        '''

        :return: расстояние X и Y в px
        '''
        distance_x = self.get_center_object()[0] - (self.img_width // 2)
        distance_y = self.get_center_object()[1] - (self.img_height // 2)
        return (distance_x, distance_y)

    def get_distance_xyn(self):
        '''

        :return: Нормализованное расстояние X и Y  от -1 до 1
        '''
        distancexy = self.get_distance_xy()
        return (distancexy[0] / (self.img_width // 2), distancexy[1] / (self.img_height // 2))

    def get_speed(self):
        '''
        :return: Скорость полета в м/c
        '''
        speed_uav = 0.001
        distancexy = self.get_distance_xy()
        return round((abs(distancexy[0]) + abs(distancexy[1])) * speed_uav, 3)

    def get_direction_uav(self):
        '''

        :return: Направление полета UAV
        '''
        distance_x, distance_y = self.get_distance_xy()
        if distance_x < 0:
            direction_x = "left"
        else:
            direction_x = "rigth"

        if distance_y < 0:
            direction_y = "up"
        else:
            direction_y = "down"

        return direction_x, direction_y

    def initialize_model(self, path):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        print(platform.system())
        if platform.system() == 'Windows':
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = onnxruntime.InferenceSession(path, providers=self.providers, opts=opts)
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        self.detect_many2one()

        # data = list(zip(self.boxes, self.scores, self.class_ids))

        # result_list = []
        #
        # for tup in data:
        #     sub_list = list(tup[0])  # Преобразовать первый элемент кортежа в список
        #     sub_list.extend([tup[1], tup[2]])  # Добавить остальные элементы кортежа в список
        #     result_list.append(sub_list)

        # self.boxes2 = Boxes(data=np.array(result_list), orig_shape=image.shape)
        # print(self.boxes2.xyxy)
        # print(self.boxes2.xywh)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def detect_many2one(self):
        max_class = 0
        max_score = 0
        self.boxes_xyxy = None
        self.score = None
        self.class_id = None
        self.center_detect = None
        for num, (class_id, box, score) in enumerate(zip(self.class_ids, self.boxes, self.scores)):
            if max_score < score:
                max_score = score
                max_class = num

        if max_score > 0:
            x1, y1, x2, y2 = self.boxes[max_class].astype(int)
            self.center_detect = int((x1 + x2) / 2), int((y1 + y2) / 2)
            self.boxes_xyxy = self.boxes[max_class].astype(int)
            self.score = max_score
            self.class_id = max_class

    def draw_navigation_uav_debug(self, image, size_text):
        return draw_navigation(drone_location=self.get_center_object(),
                               image_size=(self.img_width, self.img_height),
                               img=image, size_text=size_text)

    def draw_detection(self, image, draw_scores=True, mask_alpha=0.4, file_save_txt=None):
        return draw_detection(image, self.boxes_xyxy, self.score,
                              self.class_id, mask_alpha, file_save_txt=file_save_txt)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from imread_from_url import imread_from_url

    model_path = "../models/yolov8m.onnx"

    # Initialize YOLOv8 object detector
    yolov8_detector = YOLOv8(model_path, conf_thres=0.3, iou_thres=0.5)

    img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    img = imread_from_url(img_url)

    # Detect Objects
    yolov8_detector(img)

    # Draw detections
    combined_img = yolov8_detector.draw_detection(img)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
