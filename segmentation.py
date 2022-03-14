import cv2.cv2
from cv2 import cv2
import numpy as np
import matplotlib as mat
import math


# inital
#   provided picture of background
# background subtraction
#   - must work for glare/lighting
# fix warp perspective

def removeholes(mask):
    # fill in holes/returns mask with no holes
    mask_copy = mask.copy()

    cv2.floodFill(mask_copy, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8), (0, 0), 255)
    cv2.imshow("a",mask_copy)
    mask_copy = cv2.bitwise_not(mask_copy)
    cv2.imshow("b", mask_copy)
    outputmask = cv2.bitwise_or(mask, mask_copy)
    return outputmask

def backgroundsubtraction(frame):

    mask = np.zeros(frame.shape[:2], np.uint8)
    img = frame.copy()

    #simplify image
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(imggray, (3,3),0)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    imgdil = cv2.dilate(imgblur, kernel_rect, iterations=2)
    #imgdil = cv2.morphologyEx(imgblur,cv2.MORPH_OPEN, kernel_rect)
    image = cv2.divide(imggray, imgdil, scale=255)
    image = cv2.bitwise_not(image)

    cv2.imshow("a",image)


    #cont,_=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame,cont,-1,(0,255,0),1)
    #img = cv2.bitwise_and(frame, frame, mask=mask)
    return img, mask

def unique(frame):
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    imgdil = cv2.dilate(frame, kernel_rect, iterations=2)
    # imgdil = cv2.morphologyEx(imgblur,cv2.MORPH_OPEN, kernel_rect)
    image = cv2.divide(frame, imgdil, scale=255)
    #image = cv2.bitwise_not(image)
    return image


def kmeans(frame, K):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    kmean_image = res.reshape((img.shape))
    return kmean_image

def segmentation(frame):
    #using kmeans

    kmean_image = kmeans(frame, 2)
    cv2.imshow("kmean",kmean_image)


    imgblur = cv2.medianBlur(kmean_image, 5)

    mask = np.zeros(frame.shape[:2], np.uint8)
    for h, height in enumerate(imgblur):
        for w, width in enumerate(height):
            for c, channel in enumerate(width):
                if channel >50:
                    mask[h][w] = 255



    #mask_result = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel_rect,iterations=1)

    mask_result = mask

    img_result = cv2.bitwise_and(frame,frame, mask=mask_result)
    return img_result, mask_result

if __name__ == '__main__':
    # use color detection to differenitate pieces from background
    # green screen?
    # contours to map out shape

    imgpiece = cv2.imread("images/4blackpieces.png")
    imgpiece = cv2.resize(imgpiece, (imgpiece.shape[1]>>2, imgpiece.shape[0]>>2))
    #imgpiece = cv2.rotate(imgpiece,cv2.ROTATE_90_COUNTERCLOCKWISE)


    imgfore, mask = segmentation(imgpiece)

    #cv2.imshow("fore", imgfore)
    #cv2.imshow("mask", mask)

    cv2.waitKey(0)

    pass
