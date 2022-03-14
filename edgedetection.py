from cv2 import cv2


#
#EDGE DETECTION
#assign each piece unique identifier
#group each piece into edges,corners, or inners


def shapedetection(img,outputimg):

    contours, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        cv2.drawContours(outputimg,cont, -1, (255,0,0),3)
        pass

    pass


if __name__ == '__main__':

    imgpiece = cv2.imread("images/4pieces.png")
    imgback = cv2.imread("images/background.png")

    #imghsv = cv2.cvtColor(imgpiece, cv2.COLOR_BGR2HSV)
    #imggray = cv2.cvtColor(imgpiece, cv2.COLOR_BGR2GRAY)
    #imgblur = cv2.GaussianBlur(imggray,(5,5),0)

    #imgthresh = cv2.threshold(imgblur, 130, 255, cv2.THRESH_BINARY)[1]
    #imgcanny = cv2.Canny(imgblur,25,25)

    #imgcont = imgpiece.copy()
    #shapedetection(imgcanny,imgcont)

    #showimg = np.hstack([imgblur,imgthresh])
    #cv2.imshow("all",showimg)

    a = np.array([1,2,3])
    b = np.array([2,3,4])
    print(a-b)
    #cv2.waitKey(0)



    pass

