import cv2
import numpy as np

from re import I
def gray_image(img):
  grayed = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  return grayed


def smooth(img,x,y):
  smoothed = cv2.GaussianBlur(img, (x,y), 0)
  return smoothed


def canny(img,x,y):
  canny_img = cv2.Canny(img, x,y)
  return canny_img

def mask(img,x1,x2,y1,y2):
  height = img.shape[0]
  polygons = np.array([
  [(x1, height), (x2, height), (y1, y2)]
  ])
  mask = np.zeros_like(img)

  cv2.fillPoly(mask, polygons,(255,20,270) )
  return mask


def roi(img,mask):
    masked = cv2.bitwise_and(img,mask)
    return masked


def hough_image(i,m):
  lines = cv2.HoughLinesP(m, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
  line_image = np.zeros_like(i)
  for line in lines:
      for x1,y1,x2,y2 in line:
          cv2.line(line_image,(x1,y1),(x2,y2),(255,20,270),10)
  combo_image = cv2.addWeighted(i, 0.8, line_image, 1, 1)
  return combo_image

