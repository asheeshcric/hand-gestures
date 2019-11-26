import cv2
import imutils

from PIL import Image

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# CNN Network for the trained model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 31, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 31)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def predict_image():
    global classes
    out_img = Image.open('output.jpg')
    k = transform(out_img.convert('RGB'))
    k = k.unsqueeze(0)
    out = net(k)
    _, predicted = torch.max(out, 1)
    return classes[predicted[0]]


def average_background(image, avg_weight, background):
    # Initialize background if for the first time
    if background is None:
        background = image.copy().astype("float")
        return background

    # Compute weighted average for the background for the first few seconds
    cv2.accumulateWeighted(image, background, avg_weight)
    return background


def segment(image, background, threshold=25):
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(background.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, _ = cv2.findContours(
        thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return None
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return thresholded, segmented


def get_gray_image(cap, dims):
    # get the current frame
    grabbed, frame = cap.read()

    # Resize the frame
    frame = imutils.resize(frame, width=700)
    # Flip the camera frame (to avoid mirror image)
    frame = cv2.flip(frame, 1)
    # Create a different copy of the frame
    clone = frame.copy()

    # Crop the Region of Interest (RoI) from the entire frame
    roi = frame[dims['top']:dims['bottom'], dims['right']:dims['left']]

    # Convert RoI to grayscale and apply Gaussian blurring
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    return gray, clone


def main():
    # Initialize all the constant values at the beginning
    background = None
    avg_weight = 0.5
    dims = {'top': 10, 'bottom': 325, 'left': 690, 'right': 350}
    num_frames = 0
    classify = False
    # Font for text displayed on the image window
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get the reference to the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Handle interrupts from the user for different modes
        keypress = cv2.waitKey(1) & 0xFF
        # If 't' is pressed, start classifying the frames
        if keypress == ord('t'):
            classify = True

        # If 'z' is pressed, end the program
        if keypress == ord("z"):
            break

        # Get processed gray image along with the original frame
        gray, original_frame = get_gray_image(cap, dims)

        # To calibrate the running average model and get the background, keep looking till a threshold is reached
        # 30 allows it to calibrate for 1 sec (30 frames/sec rate)
        if num_frames < 30:
            background = average_background(gray, avg_weight, background)
        else:
            # Segment the hand portion
            hand = segment(gray, background)

            # Proceed only if the hand region is segmented
            if hand is not None:
                thresholded, segmented = hand
                # Display the segmented hand using the thresholded values for the hand
                cv2.drawContours(original_frame, [segmented + (dims['right'], dims['top'])], -1, (0, 0, 255))
                cv2.imshow("Hand Contour (B/W)", thresholded)

                if classify:
                    cv2.imwrite('output.jpg', thresholded)
                    prediction = predict_image()
                    cv2.putText(original_frame, prediction, (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # Draw a rectangle to define the Region of Interest
        cv2.rectangle(original_frame, (dims['left'], dims['top']), (dims['right'], dims['bottom']), (0, 255, 0), 2)

        # Counter for the number of frames displayed so far
        num_frames += 1

        # Display the original frame
        cv2.imshow("Original Frames", original_frame)

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # For deep learning model
    classes = ['fist', 'left', 'ok', 'right', 'straight']
    # Load trained model
    net = Net()
    net.load_state_dict(torch.load('models/gesture_model_15epoch.pth'))
    transform = transforms.Compose(
        [transforms.Scale(128),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    main()
