import cv2
import os
import time
import numpy as np


dir_name = 'cropped_images/no_gesture'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)


cap = cv2.VideoCapture(0)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Length: {}".format(length))
print("FPS: {}".format(fps))
count = 0
start = False
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    box_start = (15, 95)
    box_end = (425, 405)
    cv2.rectangle(frame, box_start, box_end, (255, 0, 0), 2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(32) == ord(' '):
        print('Writing image...')
        start = True
        start_time = time.time()

    if start:
        if time.time() - start_time > 0.05:
            count += 1
            start_time = time.time()
            cv2.imwrite('{}/image_{}.jpg'.format(dir_name, count), frame[box_start[1]+5:box_end[1]-5, box_start[0]+5:box_end[0]-5, :])
    
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 600:       
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()