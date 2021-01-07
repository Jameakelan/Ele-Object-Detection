import cv2
import numpy as np

import time
import sys
import winsound

import sendingAPI
import time
import os
import requests

frequency = 500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second


CONFIDENCE = 0.5  # Set frame to object
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
config_path = "cfg/yolov4.cfg"
weights_path = "cfg/yolov4.weights"
font_scale = 1
thickness = 1
labels = open("coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
count = 0

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture("elephant/cam11-1.mp4")
# cap = cv2.VideoCapture(0)

callAPI = False

countDown = 10

notifyCounter = 0

# def sendThread1(fileName):
#     global notifyCount
#     thread = threading.Thread(target=sendAPI, args=[fileName])
#     thread.start()

#     reqData = {'data': fileName}
#     success = sendAPI(reqData)

#     if success == 'success':
#         notifyCount = 0


def calledAPI(filename):
    print("no calling")
    if callAPI:
        print("calling")
        result = sendingAPI.notifyDection({'data': filename})
        if result['send'] == 'success':
            print(result['send'])
            # time.sleep(2)


def saveAndSendFile(fream):
    fileName = 'elephant_images{}.jpg'.format(time.time())
    path = "F:/Test-Image detection using Yolo/images"
    status = cv2.imwrite(os.path.join(fileName), fream)
    if status:
        sendingAPI.sendImageFile(fileName)


while(True):
    _, fream = cap.read()

    h, w = fream.shape[:2]
    blob = cv2.dnn.blobFromImage(
        fream, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    #print("Time took:", time_took)
    boxes, confidences, class_ids = [], [], []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the fream, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

    font_scale = 1
    thickness = 1

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw a bounding box rectangle and label on the fream
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(fream, (x, y), (x + w, y + h),
                          color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"

            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y),
                          (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = fream.copy()
            cv2.rectangle(
                overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # add opacity (transparency to the box)
            fream = cv2.addWeighted(overlay, 0.6, fream, 0.4, 0)
            # now put the text (label: confidence %)
            cv2.putText(fream, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

            if f"{labels[class_ids[i]]}" == "elephant":
               # print("elephant")
                winsound.Beep(frequency, duration)

                confidenceNum = float('{:.2f}'.format(confidences[i]))
                print(confidenceNum)
                if confidenceNum >= 0.6:

                    saveAndSendFile(fream)

                  #  if callAPI:
                    # sendingAPI.notifyDection({'data': imgBase64})
                    # thread = threading.Thread(target=calledAPI, args=[
                    #                           random.randint(0, 100)])
                    # thread.start()
                    # thread.join()
            else:
                callAPI = False

    img = cv2.resize(fream, (1200, 600))
    cv2.imshow("frame", img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
