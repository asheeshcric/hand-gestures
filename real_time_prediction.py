import cv2
import os
import time
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
from matplotlib import cm


def predict_image(image):
    global model, test_transforms
    input_image = test_transforms(image)
    input_image = input_image.view(1, 3, 224, 298)
    output = model(input_image)
    return int(torch.max(output.data, 1)[1].numpy())


def video_stream():
    cap = cv2.VideoCapture(0)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Length: {}".format(length))
    print("FPS: {}".format(fps))
    count = 0
    start_time = time.time()
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        box_start = (15, 95)
        box_end = (425, 405)
        cv2.rectangle(frame, box_start, box_end, (255, 0, 0), 2)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if time.time() - start_time > 0.5:
            # Predict every one second
            start_time = time.time()
            image = frame[box_start[1]+5:box_end[1] -
                          5, box_start[0]+5:box_end[0]-5, :]

            print(image.shape)
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            prediction = predict_image(image)
            predicted_classes = ['closed_fist', 'left',
                                 'no_gesture', 'open_palm', 'right', 'start_ok']
            # predicted_classes = {
            #     0: 'CLOSED FIST', 1: 'LEFT', 2: 'OPEN PALM', 3: 'RIGHT', 4: 'START OK'
            # }
            print('{}: {}'.format(prediction, predicted_classes[prediction]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load model
    model = torch.load('new_gesture_model.pth')
    model.eval()
    # Test transforms
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                          ])
    video_stream()
