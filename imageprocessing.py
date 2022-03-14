
from cv2 import cv2
import numpy as np
from Capture import ImgCapture

# inital
#   provided picture of background
# background subtraction
#   - must work for glare/lighting
# fix warp perspective


LOCATION = "C:/Users/danie/PycharmProjects/pythonProject/images"
class Segmentation():
    def __init__(self):
        # # intialize variables and lists
        # self.img_original = cv2.imread(LOCATION + "/superpiece.png")
        # self.img_original = self.img_original[:,20:1150]
        # cv2.imshow("orig", self.img_original)
        # self.img_gray = cv2.cvtColor( self.img_original, cv2.COLOR_BGR2GRAY)
        # self.mask, self.fore = self.binarize(self.img_original)
        pass


    def binarize(self, frame):  # differentiate pieces from background

        newimg = frame.copy()
        imggray = cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY)
        imgblur = cv2.GaussianBlur(imggray, (1, 1), 0)
        imgcanny = cv2.Canny(imgblur, 20, 80)
        cv2.imshow("sg", imgcanny)
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        kernel_dia = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
        img_mask = cv2.morphologyEx(imgcanny, cv2.MORPH_CLOSE, kernel_rect, iterations=1)

        img_mask = self.removeholes(img_mask)

        img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel_dia, iterations=1)
        img_mask = self.removenoise(img_mask)

        newimg = cv2.bitwise_and(newimg, newimg, mask=img_mask)

        return img_mask, newimg

    def colorsegment(self, frame):
        pass

    def removenoise(self, frame_binary):
        contours, _ = cv2.findContours(frame_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area=np.zeros(len(contours))
        for i,cont in enumerate(contours):
            area[i] = cv2.contourArea(cont)

        for i,cont in enumerate(contours):
            if area[i] < np.max(area)*.30:
                cv2.drawContours(frame_binary, [cont], 0, (0, 0, 0), -1)

        return frame_binary

    def kmeans(self, frame, K):
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

    def glarereduction(self, frame):
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        imgdil = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel_rect)
        image = cv2.divide(frame, imgdil, scale=255)
        image = cv2.bitwise_not(image)
        return image

    def removeholes(self, frame):
        # fill in holes/returns mask with no holes
        frame_copy = frame.copy()
        frame_copy[:,0] = 0
        frame_copy[0,:] = 0
        frame_copy[:,-1] = 0
        frame_copy[-1,:] = 0
        cv2.floodFill(frame_copy, np.zeros((frame.shape[0] + 2, frame.shape[1] + 2), np.uint8), (0, 0), 255) #BG is white
        mask_copy = cv2.bitwise_not(frame_copy) #BG IS NOW BLACK holes are now white

        result = cv2.bitwise_or(frame, mask_copy) #or mask with original
        return result

def removeshadows(frame):
    img_copy = frame.copy()
    img_copy = cv2.cvtColor(img_copy,cv2.COLOR_RGB2HSV)
    lower = np.array([18, 42, 69])
    upper = np.array([179, 255, 255])
    img_copy = cv2.inRange(img_copy, lower, upper)
    return img_copy

if __name__ == '__main__':
    Seg = Segmentation()
    cv2.imshow("mask",Seg.mask)
    cv2.imshow("fore", Seg.fore)
    cv2.waitKey(0)
    pass