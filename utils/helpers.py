import cv2
import numpy as np

CLASS_NAMES = ['bike-bicycle', 'bus-truck', 'car', 'fire', 'human', 'smoke']
COLORS = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

def preprocess(image, img_size=640):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
    return image[np.newaxis, ...]

def postprocess(outputs, conf_thresh=0.5, iou_thresh=0.5):
    outputs = outputs[0].transpose()
    boxes, scores, class_ids = [], [], []
    
    for row in outputs:
        cls_scores = row[4:4+len(CLASS_NAMES)]
        class_id = np.argmax(cls_scores)
        max_score = cls_scores[class_id]
        
        if max_score >= conf_thresh:
            cx, cy, w, h = row[:4]
            x = (cx - w/2).item()  # Convert to Python float
            y = (cy - h/2).item()
            width = w.item()
            height = h.item()
            boxes.append([x, y, width, height])
            scores.append(float(max_score))
            class_ids.append(int(class_id))

    if len(boxes) > 0:
        # Convert to list of lists with native Python floats
        boxes = [[float(x) for x in box] for box in boxes]
        scores = [float(score) for score in scores]
        
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=conf_thresh,
            nms_threshold=iou_thresh
        )
        
        if len(indices) > 0:
            boxes = [boxes[i] for i in indices.flatten()]
            scores = [scores[i] for i in indices.flatten()]
            class_ids = [class_ids[i] for i in indices.flatten()]

    return boxes, scores, class_ids