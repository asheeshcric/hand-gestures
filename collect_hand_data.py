import cv2
import imutils
import os
import time


def average_background(image, avg_weight, background):
    # Initialize background if for the first time
    if background is None:
        background = image.copy().astype("float")
        return background

    # Compute weighted average for the background for the first few seconds
    cv2.accumulateWeighted(image, background, avg_weight)
    return background


def segment(image, background, threshold=25):
    # Finding the absolute difference between image and the background
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
    # Retrieve the current frame
    _, frame = cap.read()

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


def collect_images():
    # Initialize all the constant values at the beginning
    background = None
    avg_weight = 0.5
    dims = {'top': 10, 'bottom': 325, 'left': 690, 'right': 350}
    num_frames = 0

    # Details for storage of image data
    image_class = 'ok'
    store_path = f'./datasets/gesture_contours/{image_class}'
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    img_count = len(os.listdir(store_path))
    this_count = 1
    store = False
    start_time = time.time()

    # Capture frames from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Handle interrupts from the user for different modes
        keypress = cv2.waitKey(1) & 0xFF
        # If 't' is pressed, start classifying the frames
        if keypress == ord('t'):
            print('Collecting images...')
            store = True

        # If 'z' is pressed, end the program
        if keypress == ord("z") or this_count % 600 == 0:
            print('Task completed!')
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

                if store and time.time() - start_time > 0.05:
                    # Store images for the particular class at certain intervals
                    img_count += 1
                    this_count += 1
                    cv2.imwrite(f'{store_path}/image_{img_count}.png', thresholded)

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
    collect_images()
