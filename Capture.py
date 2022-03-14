from cv2 import cv2


class ImgCapture:
    def __init__(self,location):
        self.num = 0
        self.location = location
        pass

    def getImageManual(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 1280) #width
        cap.set(4, 720) #height
        #brightness-10
        #frame rate - 5


        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #1920
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 1080
        #CHECK RESOLUTION
        #w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        #h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #print(w,h)
        while True:
            success, img = cap.read()
            cv2.imshow("Video", img)
            input = cv2.waitKey(1)
            if input & 0xFF == ord('p'):
                self.num += 1
                cv2.imshow("piece", img)
                cv2.imwrite(self.location + f"/piece{self.num}.png", img)

            if input & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    LOCATION = "C:/Users/danie/PycharmProjects/pythonProject/images"
    cap = ImgCapture(LOCATION)
    cap.getImageManual()

    pass