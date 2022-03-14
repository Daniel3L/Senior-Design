from cv2 import cv2
from imageprocessing import Segmentation

class ImgCapture:
    def __init__(self,location):
        self.num = 1
        self.Seg = Segmentation()
        self.location = location


    def getImageManual(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 1280) #width
        cap.set(4, 720) #height
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #1920
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 1080
        flag = 0
        while True:
            success, img = cap.read()
            img = img[150:650,450:1000]
            cv2.imshow("Video", img)
            binimg, fore = self.Seg.binarize(img)
            cv2.imshow("Binarize", binimg)
            #cv2.imshow("Foreground", fore)
            input = cv2.waitKey(1)
            if input & 0xFF == ord('p'):
                cv2.imshow("piece", img)
                cv2.imshow("Bin", binimg)
                cv2.imwrite(self.location + f"/piece{self.num}.png", img)

            elif input & 0xFF == ord('q'):
                flag = 1
                break
            elif input & 0xFF == ord('n'):
                self.num += 1
                cv2.destroyWindow("piece")
                cv2.destroyWindow("Bin")
                print(f"taking image {self.num}")
            elif input & 0xFF == ord('c'):
                print("CANCELED")
                break

        cap.release()
        cv2.destroyAllWindows()
        return flag

    def prompt(self):
        print("Constaints: ")
        print("1. Puzzle is standard rectangular shape with 4 90-degree corner pieces")
        print("2. All puzzle pieces are adequately spaced apart (no overlapping)")
        print("")
        print("P = snapshot")
        print("Q = quit and solve puzzle snapshot")
        print("N = new snapshot")
        print("C = cancel")


if __name__ == '__main__':
    LOCATION = "C:/Users/danie/PycharmProjects/pythonProject/images"
    cap = ImgCapture(LOCATION)
    cap.getImageManual()

