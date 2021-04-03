import cv2
import numpy as np

cap = cv2.VideoCapture(0)

result = True
while result:
    _,img = cap.read()
    cv2.imwrite("img.png", img)
    result = False

while (cap.isOpened()):
    _,frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_red = np.array([0, 0, 39])
    u_red = np.array([0,0,0])
    mask_1 = cv2.inRange(hsv, l_red, u_red)

    lower_red = np.array([0,0,90])
    upper_red = np.array([0,0,100])
    mask_2 = cv2.inRange(hsv, lower_red, upper_red)

    mask_1 = mask_1 + mask_2

    #Open and expand the image where there is mask 1 (color)
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    #Selecting only the part that does not have mask one and saving in mask 2
    mask_2 = cv2.bitwise_not(mask_1)

    #Keeping only the part of the images without the red color 
    #(or any other color you may choose)
    res_1 = cv2.bitwise_and(frame, frame, mask=mask_2)

    #Keeping only the part of the images with the red color
    #(or any other color you may choose)
    res_2 = cv2.bitwise_and(img, img, mask=mask_1)

    #Generating the final output by merging res_1 and res_2
    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)

    cv2.imshow("result", final_output)

    # break the loop on pressing esc
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        result = False
        break