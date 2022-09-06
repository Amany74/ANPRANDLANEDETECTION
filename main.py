# Import for the app
import cv2
import streamlit as st
import numpy as np
import joblib
import numpy as np
import matplotlib.pyplot as plt
import imutils
import easyocr
import os
from lane import *
#ANPR
# Step one convert image to grayscale
def convert_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

#step 2 apply smoothing filter on a grayed image
def convert_bi(img):
    bilateral = cv2.bilateralFilter(convert_gray(img), 20, 17, 17)
    return bilateral

# Step 3 find edges with canny
def convert_canny(img):
    edged = cv2.Canny(convert_bi(img), 30, 200)
    return edged

# step 4 find contours
def convert_contours(img):
    keypoints = cv2.findContours(convert_canny(img).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    return contours

# step 5 get the number using easyocr
def detect_plate(img):
    contours = convert_contours(img)
    # Specify the location of the plate
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 30, True)
        if (len(approx) == 4):
            location = approx
            break
    gray = convert_gray(img)
    mask = np.zeros(gray.shape,np.uint8)
    new_img = cv2.drawContours(mask,[location],0,255,-1)
    new_image = cv2.bitwise_and(img,img,mask=mask)

    # crop it
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_img = gray[x1:x2 + 1, y1:y2 + 1]

    st.write("getting the plate number ....")
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_img)
    text = result[0][-2]
    font = cv2.FONT_HERSHEY_TRIPLEX
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=2,
                      color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
    return res


# LANE DETECTIION
def detect_lane(img2):
    # gray image
    gray_img = gray_image(img2)
    # smooth
    smooth_img = smooth(gray_img, 11, 11)
    # canny
    canny_img = canny(smooth_img, 50, 150)
    # add mask
    masked = mask(canny_img, 0, 1850, 900, 350)
    img_roi = roi(canny_img, masked)
    # with lines
    final = hough_image(img2, img_roi)
    return final



if __name__ == "__main__":
    st.title("Welcome , ANPR and Lane Detection")
    st.subheader('Detect your license plate number simply or Lane detection ')
    uploaded_file = st.file_uploader("Choose a image file", type=['jpg','png','jpeg'])

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # display the image
        st.image(opencv_image, channels="BGR")
        if st.button('Detect Plate'):
            result = detect_plate(opencv_image)
            st.write('result: \n')
            st.image(result)
        if st.button('Lane detect'):
            result = detect_lane(opencv_image)
            st.write('result: \n')
            st.image(result)
