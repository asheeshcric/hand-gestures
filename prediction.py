import cv2
import keyboard
import numpy as np
import math
import os
from PIL import Image

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def predict_image():
    global classes
    out_img = Image.open('output.jpg')
    k = transform(out_img.convert('RGB'))
    k = k.unsqueeze(0)
    out = net(k)
    _, predicted = torch.max(out, 1)
    return classes[predicted[0]]


def control_keyboard(prediction):
    controls = {
        'open_palm': '',
        'ok': '',
        'left': '',
        'right': '',

    }


# CNN Network for the trained model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def display_output(frame, arearatio, areacnt, l, prediction):
    # print corresponding gestures which are in their ranges
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(f'{prediction}: ({arearatio}, {areacnt}, {l})')
    if l == 1 and prediction != 'fist':
        cv2.putText(frame, 'NO GESTURE', (0, 50),
                    font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        if prediction == 'fist':
            cv2.putText(frame, 'FIST', (0, 50), font, 2,
                        (0, 0, 255), 3, cv2.LINE_AA)

        elif prediction == 'left':

            if arearatio < 27:
                cv2.putText(frame, 'LEFT', (0, 50), font, 2,
                            (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'ok', (0, 50), font, 2,
                            (0, 0, 255), 3, cv2.LINE_AA)

        elif prediction == 'ok':
            cv2.putText(frame, 'OK', (0, 50), font, 2,
                        (0, 0, 255), 3, cv2.LINE_AA)

        elif prediction == 'open_palm':
            cv2.putText(frame, 'OPEN PALM', (0, 50), font, 2,
                        (0, 0, 255), 3, cv2.LINE_AA)

        elif prediction == 'right':
            cv2.putText(frame, 'RIGHT', (0, 50), font,
                        2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'NO GESTURE', (10, 50), font,
                        2, (0, 0, 255), 3, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)

    start = False
    dir_name = 'datasets/contours/open_palm'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    count = len(os.listdir(dir_name))

    while 1:

        try:  # an error comes if it does not find anything in window as it cannot find contour of max area
            # therefore this try error statement

            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            kernel = np.ones((3, 3), np.uint8)

            # define region of interest
            box_start = (250, 20)
            box_end = (600, 370)
            roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]

            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 0)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # define range of skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            # extract skin colur imagw
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # extrapolate the hand to fill dark spots within
            mask = cv2.dilate(mask, kernel, iterations=4)

            # blur the image
            mask = cv2.GaussianBlur(mask, (5, 5), 100)

            # find contours
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # find contour of max area(hand)
            cnt = max(contours, key=lambda x: cv2.contourArea(x))

            # approx the contour a little
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # make convex hull around hand
            hull = cv2.convexHull(cnt)

            # define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)

            # find the percentage of area not covered by hand in convex hull
            arearatio = ((areahull - areacnt) / areacnt) * 100

            # find the defects in convex hull with respect to hand
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)

            # l = no. of defects
            l = 0

            # code for finding no. of defects due to fingers
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100, 180)

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                # distance between point and convex hull
                d = (2 * ar) / a

                # apply cosine rule here
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d > 30:
                    l += 1
                    cv2.circle(roi, far, 3, [255, 0, 0], -1)

                # draw lines around hand
                cv2.line(roi, start, end, [0, 255, 0], 2)

            l += 1

            # Need to convert "mask" numpy array to a pillow image
            cv2.imwrite('output.jpg', mask)
            prediction = predict_image()

            display_output(frame, areacnt=areacnt, arearatio=arearatio, l=l, prediction=prediction)
            keyboard.write('keyboard')

            # show the windows
            cv2.imshow('mask', mask)
            cv2.imshow('frame', frame)

        except Exception as error:
            print(error)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    classes = ['fist', 'left', 'ok', 'open_palm', 'right']
    # Load trained model
    net = Net()
    net.load_state_dict(torch.load('models/contour_model.pth'))
    transform = transforms.Compose(
        [transforms.Scale(128),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    main()
