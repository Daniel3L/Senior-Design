from PuzzleSolver2 import PuzzleSolver
from Capture2 import ImgCapture
import cv2 as cv2
import numpy as np
# class Solver:
#     def __init__(self,location):
#         self.Cap = ImgCapture(location)
#         self.Cap.prompt()
#         a = self.Cap.getImageManual()
#         if a == 1:
#             puzzle = PuzzleSolver(cv2.imread(location + f"/piece{0}.png"), 1)

def Solver(location, display=0):
    Cap = ImgCapture(location)
    Cap.prompt()
    flag = Cap.getImageManual()
    if flag:
        puzzle = PuzzleSolver(cv2.imread(location + f"/piece{0}.png"), display)
        return puzzle.solution_matrix, puzzle.piece

    return np.array([]), None


if __name__ == '__main__':
    location = "C:/Users/danie/PycharmProjects/pythonProject/images"
    Solver(location, 1)
    cv2.waitKey(0)
    pass