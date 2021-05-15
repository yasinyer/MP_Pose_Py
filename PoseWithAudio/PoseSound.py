import cv2
import queue
import threading
import time
import numpy as np
import mediapipe as mp
from threading import Thread
from playsound import playsound
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.dataset import transform_images
import multiprocessing
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

size = 416
classes = "coco/coco.names"
num_classes = 80
yolo = YoloV3Tiny()

yolo.load_weights("coco/yolov3-tiny.tf")

class_names = [c.strip() for c in open(classes, "r").readlines()]


def play():
    while True:
        sound = q.get()
        if sound is None:
            break
        playsound(sound)
        time.sleep(1)


q = queue.Queue()
personQ = queue.Queue()
personQResult = queue.Queue()
piano = "g-3.mp3"
pianoThread = Thread(target=play, daemon=True).start()


def process():
    while True:
        p = personQ.get()
        print(p)
        if p is None:
            break
        personQResult.put(pose.process(p))


PersonThread = Thread(target=process, daemon=True).start()

# For webcam input:
cap = cv2.VideoCapture(0)


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def getPersonImages(img, outputs, class_names):
    images = []
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        if class_names[int(classes[i])] == "person":
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            images.append(cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2))
    return images


def start():
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        img = tf.expand_dims(image, 0)
        img = transform_images(img, size)
        boxes, scores, classes, nums = yolo.predict(img)
        # img = draw_outputs(image, (boxes, scores, classes, nums), class_names)
        [personQ.put(x) for x in getPersonImages(image, (boxes, scores, classes, nums), class_names)]
        if personQResult.qsize() > 1:
            results = personQResult.get()
            if results:
                noseX = results.pose_landmarks.landmark[0].x
                if noseX > 0.6:
                    q.put(piano)

                if 0.6 > noseX > 0.4:
                    q.queue.clear()

                    # if noseX < 0.4:
                    #     T = Thread(target=play, args=(sounds["piano"],), daemon=True)
                    #     T.start()

            # Draw the pose annotation on the image.
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Nose X Piano', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('----------')
    print("starting...")
    print('----------')
    start()
