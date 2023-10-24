import cv2
import numpy as np

class_names = ['Helicopter']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))
font = cv2.FONT_HERSHEY_COMPLEX


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(x):
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            # print(new_width)
            return scale / 10
    return 1


def draw_navigation(drone_location, image_size, img=None, size_text=12):
    if drone_location:
        # Размеры изображения
        image_width, image_height = image_size

        # Расчет центра изображения
        image_center_x = image_width // 2
        image_center_y = image_height // 2

        # Текущая позиция БПЛА
        drone_x, drone_y = drone_location

        # Расчет расстояния до центра изображения
        distance_x = drone_x - image_center_x
        distance_y = drone_y - image_center_y

        # Вывод расстояния до центра в пикселях
        # print(
        # f"Расстояние до центра изображения: {abs(distance_x)} пикселя(-ей) по горизонтали, {abs(distance_y)} пикселя(-ей) по вертикали")

        # Расчет нормализованных значений
        normalized_x = distance_x / image_center_x
        normalized_y = distance_y / image_center_y

        # Вывод нормализованных значений
        # print(f"Нормализованное значение (от -1 до 1): по горизонтали: {normalized_x}, по вертикали: {normalized_y}")

        # Определение направления движения для центрирования
        if distance_x < 0:
            direction_x = "left"
        else:
            direction_x = "rigth"

        if distance_y < 0:
            direction_y = "up"
        else:
            direction_y = "down"

        Color_text = (217, 0, 255)
        text1 = f'distance UAV to center frame: '
        text2 = f'{round(abs(distance_x), 3)} px horizontal. fly to {direction_x} // normalized value: {round(normalized_x, 3)}'
        text3 = f'{round(abs(distance_y), 3)} px vertical. fly to {direction_y} // normalized value: {round(normalized_y, 3)}'

        # print(len(text2))

        text_positionX = image_width * 0.5

        scale_text = get_optimal_font_scale('*'*58, text_positionX)
        cv2.putText(img, text1, (int(image_width*0.2), 30), font, scale_text, Color_text, 1, cv2.LINE_AA)
        cv2.putText(img, text2, (int(image_width*0.2), 60), font, scale_text, Color_text, 1, cv2.LINE_AA)
        cv2.putText(img, text3, (int(image_width*0.2), 90), font, scale_text, Color_text, 1, cv2.LINE_AA)
        speed_uav = 0.001
        cv2.putText(img, f'Speed: {round((abs(distance_x) + abs(distance_y)) * speed_uav, 3)} m/c',
                    (int(image_width*0.8), 30), font, scale_text, Color_text, 1, cv2.LINE_AA)

        cv2.line(img, (image_center_x, image_center_y), drone_location, (200, 0, 0), 2)
        cv2.circle(img, (image_center_x, image_center_y), 15, (0, 255, 12), 2)

        # text = f"Двигаться {'влево' if distance_x < 0 else 'вправо'} {'по горизонтали' if abs(distance_x) > 10 else ''}"
        # Вывод направления движения
        # print(f"{text}")
    return img


def draw_detection(image, boxes_xyxy, score, class_id, mask_alpha=0.3, file_save_txt=None):
    det_img = image.copy()
    img_height, img_width = image.shape[:2]
    if (boxes_xyxy is not None) and (score > 0):

        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # center_frame = (img_width // 2, img_height // 2)

        det_img = draw_masks(det_img, np.array([boxes_xyxy]), np.array([class_id]), mask_alpha)
        color = colors[class_id]
        draw_box(det_img, boxes_xyxy, color)
        label = class_names[class_id]
        sc = str(score.round(2))
        caption = f'{label} {sc}'
        draw_text(det_img, caption, boxes_xyxy, color, font_size, text_thickness)
        x1, y1, x2, y2 = boxes_xyxy.astype(int)

        # print(x1, y1, x2, y2)
        size_circle = 0
        x2minusx1 = (x2 - x1) // 2
        y2minusy1 = (y2 - y1) // 2

        if x2minusx1 > y2minusy1:
            size_circle = (x2minusx1 - y2minusy1) // 3
        else:
            size_circle = (y2minusy1 - x2minusx1) // 3

        center = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.putText(det_img, f'{label}, center: {center} ', (30, img_height - 30), font, 1, (0, 255, 50), 2,
                    cv2.LINE_AA)
        cv2.circle(det_img, center, size_circle, (0, 255, 12), 2)
        # calculate_position_drone(det_img, center, (img_width, img_height))

        text = f'center: {center}, box xyxy: {boxes_xyxy.astype(int)}, detect: {caption}'

        if file_save_txt:
            with open(file_save_txt, 'a', encoding='utf-8') as f:
                f.write(text + "\n")
    else:
        cv2.putText(det_img, f'::NO DETECTION::', (30, img_height - 30), font, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

    return det_img


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness,
                       cv2.LINE_AA)


def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
