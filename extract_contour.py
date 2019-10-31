import cv2
import os

source_dir = 'gesture_images'
dest_dir = 'hand_contours'

"""
This function is to process the image and extract hand contour from it
"""
def process_img(img, threshold=25):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    bg = gray.copy().astype("float")
    diff = cv2.absdiff(bg.astype("uint8"), gray)

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    return thresholded


"""
The below part is to process all the images present in the current dataset
"""

def main():
    for class_dir in os.listdir(source_dir):
        path = os.path.join(source_dir, class_dir)
        for image_name in os.listdir(path):
            image_path = os.path.join(path, image_name)
            dest_path = image_path.replace(source_dir, dest_dir)
            # Check if the dest path exists or not and create dirs accordingly
            dest_dir = dest_path.replace(image_name, '')
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            img = cv2.imread(image_path)
            processed_img = process_img(img)

            cv2.imwrite(dest_path, processed_img)
            break
        break


if __name__ == '__main__':
    main()