import cv2
import random
import numpy as np
from itertools import product
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros

cap = cv2.VideoCapture(1)

# Load Yolo
net = cv2.dnn.readNet("5kelas_final.weights", "5kelas_conf.cfg")
classes = []
with open("5kelas.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
probability_minimum = 0.5
threshold = 0.5
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')  # Generate Random Color
font = cv2.FONT_HERSHEY_SIMPLEX

kernel = cv2.getGaussianKernel(3, 0)

num_detection = 0
num_proses = 0

num_detection1 = 0

# Mengambil frame dari kamera secara terus-menerus
while True:
    # Membaca frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (720, 500))
    #frame = cv2.resize(frame, (640, 360))
    num_proses = num_proses + 1

    # Menambahkan noise pada frame
    noise_level = 30
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            frame[i, j] = frame[i, j] + random.randint(-noise_level, noise_level)

    filtered_frame = cv2.filter2D(frame, -1, kernel)

    height, width, channels = filtered_frame.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(filtered_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > probability_minimum:  # Confidence Level -> Accuracy
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, probability_minimum, threshold)

    for i in range(len(boxes)):
        if i in indexes:
            num_detection = num_detection + 1
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            # color = colors[i]
            color = (0, 255, 255)
            cv2.rectangle(filtered_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(filtered_frame, label + " " + str(round(confidence, 2)), (x, y - 20), font, 1, color, 2)
            cv2.putText(filtered_frame, "Hasil Deteksi "+str(num_detection), (x, y +350), font, 1, color, 2)
            cv2.putText(filtered_frame, "Percobaan deteksi "+str(num_proses), (x , y+300), font, 1, color, 2)
            print("Percobaan deteksi dengan GF: "+str(num_proses))
            print("Hasil Deteksi dengan GF: "+str(num_detection))
    #cv2.imshow("Gaussian Filter", filtered_frame)
    cv2.imshow("Gaussian Filter", filtered_frame)

    height1, width1, channels1 = frame.shape
    # Detecting objects
    blob1 = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob1)
    outs1 = net.forward(output_layers)

    # Showing informations on the screen
    class_ids1 = []
    confidences1 = []
    boxes1 = []
    for out1 in outs1:
        for detection1 in out1:
            scores1 = detection1[5:]
            class_id1 = np.argmax(scores1)
            confidence1 = scores1[class_id1]
            if confidence1 > probability_minimum:  # Confidence Level -> Accuracy
                # Object detected
                center_x1 = int(detection1[0] * width1)
                center_y1 = int(detection1[1] * height1)
                w1 = int(detection1[2] * width1)
                h1 = int(detection1[3] * height1)

                # Rectangle coordinates
                x1 = int(center_x1 - w1 / 2)
                y1 = int(center_y1 - h1 / 2)
                boxes1.append([x1, y1, w1, h1])
                confidences1.append(float(confidence1))
                class_ids1.append(class_id1)

    indexes1 = cv2.dnn.NMSBoxes(boxes1, confidences1, probability_minimum, threshold)

    for i in range(len(boxes1)):
        if i in indexes1:
            num_detection1 = num_detection1 + 1
            x1, y1, w1, h1 = boxes1[i]
            label1 = str(classes[class_ids1[i]])
            confidence1 = confidences1[i]
            # color = colors[i]
            color1 = (255, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), color1, 2)
            cv2.putText(frame, label1 + " " + str(round(confidence1, 2)), (x1, y1 - 20), font, 1, color1, 2)
            cv2.putText(frame, "Hasil deteksi "+str(num_detection1), (x1, y1 + 350), font, 1, color1, 2)
            cv2.putText(frame, "Percobaan deteksi "+str(num_proses), (x1 , y1+300), font, 1, color1, 2)


    cv2.imshow("YOLO ONLY", frame)
    print("Percobaan deteksi tanpa GF: " + str(num_proses))
    print("Hasil Deteksi tanpa GF: " + str(num_detection))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup kamera dan menghapus semua window
cap.release()
cv2.destroyAllWindows()