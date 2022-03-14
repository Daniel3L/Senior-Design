from cv2 import cv2
import numpy as np
from Capture2 import ImgCapture
from imageprocessing import Segmentation
import math as m
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

LOCATION = "C:/Users/danie/PycharmProjects/pythonProject/images"



def edistance(ref, location):
    return m.sqrt((ref[0] - location[0]) ** 2 + (ref[1] - location[1]) ** 2)

def cart2pol(x, y):
    theta = m.atan2(y, x)
    rho = m.sqrt(x * x + y * y)
    return rho, theta


def dot(v, w):
    x, y = v
    X, Y = w
    return x * X + y * Y


def length(v):
    x, y = v
    return m.sqrt(x * x + y * y)


def vector(b, e):
    x, y = b
    X, Y = e
    return X - x, Y - y


def unit(v):
    x, y = v
    mag = length(v)
    return x / mag, y / mag


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    x, y = v
    return x * sc, y * sc


def add(v, w):
    x, y = v
    X, Y = w
    return x + X, y + Y


def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return dist, nearest


def possiblewidths(num, edge_length):
    width_list = []
    for k in range(num)[2:num]:
        if num % k == 0:
            height = (num / k) - 2
            width = k - 2
            if 2 * (width + height) == edge_length:
                width_list.append(k)

    return width_list

def rotate_image(mat, angle):
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

class PuzzleSolver:
    def __init__(self, location, flag = 0):
        self.Seg = Segmentation()
        self.Cap = ImgCapture(location)
        self.Cap.prompt()
        video_flag = self.Cap.getImageManual()
        if not video_flag:
            return
        self.orig = cv2.imread(location + f"/piece{self.Cap.num}.png")

        self.binary, self.foreground = self.Seg.binarize(self.orig)
        self.foreimg = cv2.cvtColor(self.foreground.astype(np.float32) / 255, cv2.COLOR_BGR2Luv)

        # TEST IMAGES---------
        self.image_test = self.orig.copy()
        # self.foreimg = self.foreground.copy()
        self.foreimg_test = cv2.cvtColor(self.foreground.astype(np.float32) / 255, cv2.COLOR_BGR2Luv)
        self.labelimg = self.orig.copy()
        self.binary_test = self.binary.copy()

        #ANSWER IMAGE
        self.answer = np.array([])
        self.cornerflag = 0
        # ------------------
        # 3d,3d,tuple,2d,scalar,[top left bottom right] 4x2
        # corners leftmost -> clockwise
        # straight = 0 head= 1 hole = -1
        # top left bot right
        self.type = {"edge": [], "corner": [], "middle": [], "square": [], "unknown": []}
        self.piece = {1: {"cont": np.array([]), "hull": np.array([]), "centroid": (0, 0),
                          "corners": np.zeros((4, 2)), "rotation": 0., "sides": np.zeros((4, 3)),
                          "sidecont": np.array([np.array([]), np.array([]), np.array([]), np.array([])],
                                               dtype=object)}}

        self.labeling()

        print(self.type)
        self.solution_matrix = np.array([])
        if len(self.type["corner"]) == 4 and not self.type["square"] and not self.type["unknown"]:
            self.solution_matrix = self.solve()
            if self.solution_matrix.size != 0:
                display, show_flag = self.placeheadnhole(self.cornerflag)
                if show_flag:
                    self.answer = self.show_solved()
                else:
                    self.answer = display
                cv2.imshow("answer", self.answer)
                cv2.waitKey(0)
        else:
            self.solution_matrix = np.array([])
            print("error in labeling")

        if flag:
            self.display()


    def solve(self, start=0, iteration=1):

        completed_puzzle = np.array([])  # output 2d
        total_list = []  # list of all unmatched pieces
        open_list = self.type["corner"].copy()  # to explore
        closed_list = []  # explored/matched
        parent = {open_list[start]: None}

        for i in self.piece:
            total_list.append(i)

        # STARTING PUZZLE PIECE **TOP LEFT CORNER**
        side = self.piece[open_list[start]]["sides"][:, 0]
        if side[0] == 0 and side[1] == 0:  # top/left
            pass
        elif side[1] == 0 and side[2] == 0:  # left/bot
            self.piece[open_list[start]]["rotation"] -= 3 * np.pi / 2
            self.piece[open_list[start]]["sides"] = np.roll(self.piece[open_list[0]]["sides"], -1, axis=0)
            self.piece[open_list[start]]["sidecont"] = np.roll(self.piece[open_list[0]]["sidecont"], -1, axis=0)
        elif side[2] == 0 and side[3] == 0:  # bot/right
            self.piece[open_list[start]]["rotation"] -= np.pi
            self.piece[open_list[start]]["sides"] = np.roll(self.piece[open_list[0]]["sides"], 2, axis=0)
            self.piece[open_list[start]]["sidecont"] = np.roll(self.piece[open_list[0]]["sidecont"], 2, axis=0)
        else:  # 3/0 right/top
            self.piece[open_list[start]]["rotation"] -= np.pi / 2
            self.piece[open_list[start]]["sides"] = np.roll(self.piece[open_list[0]]["sides"], 1, axis=0)
            self.piece[open_list[start]]["sidecont"] = np.roll(self.piece[open_list[0]]["sidecont"], 1, axis=0)

        closed_list.append(open_list[start])
        total_list.remove(open_list[start])
        open_list.clear()

        # edges have 3 pairing 2 sides/edges 1 middle
        # corners have 2 pairings which are sides
        # middles have 4 pairings which cannot be corner

        # SOLVING ALGORITHM
        ref_side = 3
        puzzle_width = 0
        puzzle_height = 0
        match_count = 0  # how many pieces matched (middle)

        width_count = 1  # starts at one since first corner already matched
        width_list = possiblewidths(len(self.piece), len(self.type["edge"]))
        w_count = 1
        print(width_list)
        while total_list:
            if not width_list:
                print("NO POSSIBLE WIDTHS")
                break
            # MATCHING BORDER
            if any(x in total_list for x in self.type["edge"]) or any(x in total_list for x in self.type["corner"]):
                # open_list = list(set(total_list).symmetric_difference(self.type["middle"]))
                edge_list = list(set(total_list).intersection(self.type["edge"]))
                corner_list = list(set(total_list).intersection(self.type["corner"]))

                if ref_side != 0: ref_side = len(corner_list)  # ONLY WORKS IF 4 CORNERS IDENTIFIED

                if completed_puzzle.size == 0:  # if puzzle dimensions not known
                    # open_list = edge_list + corner_list
                    if width_count + 1 >= width_list[-1]:
                        open_list = corner_list
                    elif width_count + 1 < width_list[0]:
                        open_list = edge_list
                    else:
                        if width_count + 1 == width_list[w_count]:
                            open_list = corner_list + edge_list
                            w_count += 1
                        else:
                            open_list = edge_list
                    width_count += 1
                elif ref_side != 3:  # puzzle dimensions known and error check
                    if ref_side % 2 == 0:  # if ref_side is even - height
                        temp_val = 1 if ref_side == 2 else 2
                        if m.fabs(len(closed_list) + 1 - puzzle_width) + temp_val == puzzle_height * temp_val:
                            open_list = corner_list
                        else:
                            open_list = edge_list
                    else:  # odd - widths
                        temp_val = 2
                        if m.fabs(len(closed_list) + 1 - puzzle_height) + temp_val == puzzle_width * temp_val:
                            open_list = corner_list
                        else:
                            open_list = edge_list
                else:
                    print("border error ,ref_side = 3")
                    break

                # check here if the first width is getting too long or too short
                best_score = 0
                border_name = 0  # border of 0 means no match /no 0s piece
                border_side = 0  # random

                t1, a1, c1 = self.piece[closed_list[-1]]["sides"][ref_side]  # get info on the last matched piece
                for node in open_list:  # piece name

                    # temp_side_list = np.vstack([self.piece[node]["sides"],self.piece[node]["sides"][0]])
                    # for each side of the piece (4 iterations)
                    for indx, side in enumerate(self.piece[node]["sides"]):
                        t2, a2, c2 = side

                        if t1 == 0:
                            print("side error")
                            break

                        if t1 == -t2:  # potential pair if heads and hole
                            # potential pair only if side on the right is flat
                            side_location = indx - 1 if indx != 0 else 3
                            if self.piece[node]["sides"][side_location][0] == 0:
                                # METRIC GOES HERE FINDING BEST MATCH
                                overlap_diff = m.fabs(a1 - a2)
                                color_diff = self.colormetric(node, indx, closed_list[-1], ref_side)
                                score = self.totalmetric(overlap_diff, color_diff)
                                if score > best_score:
                                    best_score = score
                                    border_name = node
                                    border_side = indx

                if border_name != 0:  # if there is a match

                    self.rotate(border_name, border_side, ref_side)  # ROTATE AND ADJUST SIDES FOR MATCH
                    closed_list.append(border_name)  # insert into matched list
                    total_list.remove(border_name)  # remove from unmatched list

                    # if completed_puzzle.size == 0:
                    #    width_count += 1
                    # if first corner detected
                    if border_name in self.type["corner"] and len(corner_list) == len(self.type["corner"]) - 1:  # == 3
                        # first corner detected
                        puzzle_width = len(closed_list)
                        side_width = len(closed_list) - 2
                        if len(self.piece) % puzzle_width == 0:  # if integer/ multiple check
                            puzzle_height = int(len(self.piece) / puzzle_width)
                            side_height = (len(self.piece) / puzzle_width) - 2
                            if 2 * (side_width + side_height) == len(self.type["edge"]):  # perimeter check
                                completed_puzzle = np.zeros((puzzle_height, puzzle_width))
                                print(f"DIMENSIONS {puzzle_height} by {puzzle_width}")
                            else:
                                print(f"INCORRECT PERIMETER {puzzle_width},{len(self.piece)}")
                                break
                        else:
                            print(f"INCORRECT AREA {puzzle_width},{len(self.piece)}")
                            break

                else:  # *NO MATCHES BORDER_NAME = 0
                    print("no matches border")
                    break


            # *******************MATCHING MIDDLES**************************
            elif any(x in total_list for x in self.type["middle"]):
                open_list = list(set(total_list).intersection(self.type["middle"]))
                # match_count == len(self.type["middle"]-len(open_list)
                best_score = 0
                border_name = 0  # border of 0 means no match /no 0s piece
                border_side = 0  # random
                ref_side = 3  # right
                # last matched piece
                last_border = len(self.type["edge"]) + len(self.type["corner"]) - 1
                right_matched = puzzle_width - 1
                if match_count % (puzzle_width - 2) == 0:  # vertical/ left side border pieces
                    asd = match_count // (puzzle_width - 2)
                    last_matched = closed_list[last_border - int(asd)]
                else:
                    last_matched = closed_list[-1]

                if match_count >= (puzzle_width - 2):  # horizontal/top pieces
                    top_matched = closed_list[last_border + temp_count]
                    temp_count += 1
                else:
                    temp_count = 1
                    top_matched = closed_list[match_count + 1]

                if match_count + 1 % (puzzle_width - 2) == 0:  # right side border pieces
                    asd = match_count + 1 // (puzzle_width - 2)
                    t5, a5, c5 = self.piece[closed_list[right_matched + int(asd)]]["sides"][1]  # left side
                else:
                    t5 = 0

                if len(open_list) <= puzzle_width-2:  # bot side border pieces
                    ndr = puzzle_width-2 - len(open_list)
                    t7, a7, c7 = self.piece[closed_list[last_border - (puzzle_height - 2) - ndr - 1]]["sides"][0]  # top side

                else:
                    t7 = 0

                t1, a1, c1 = self.piece[last_matched]["sides"][3]  # info on the last matched piece, 3-rightside
                # extract more piece info (top piece bottom)
                t3, a3, c3 = self.piece[top_matched]["sides"][2]  # top piece info, 2-botside
                for node in open_list:

                    for indx, side in enumerate(self.piece[node]["sides"]):
                        t2, a2, c2 = side
                        # extract more side info (top/right side)
                        side_location = indx - 1 if indx != 0 else 3
                        t4, a4, c4 = self.piece[node]["sides"][side_location]
                        side_location2 = indx + 2 if indx == 0 or indx == 1 else indx - 2
                        t6, a6, c6 = self.piece[node]["sides"][side_location2]
                        side_location3 = indx + 1 if indx != 3 else 0
                        t8, a8, c8 = self.piece[node]["sides"][side_location3]
                        t56 = 1
                        t78 = 1
                        if t5 and t5 != -t6:
                            t56 = 0
                        if t7 and t7 != -t8:
                            t78 = 0
                        if t1 == -t2 and t3 == -t4 and t56 and t78:  # potential pair if heads and hole
                            # METRIC GOES HERE FINDING BEST MATCH
                            overlap_diff = m.fabs(a1 - a2)
                            overlap_diff2 = m.fabs(a3 - a4)
                            color_diff = self.colormetric(node, indx, last_matched, 3)
                            color_diff2 = self.colormetric(node, side_location, top_matched, 2)
                            score1 = self.totalmetric(overlap_diff, color_diff)
                            score2 = self.totalmetric(overlap_diff2, color_diff2)
                            score = (score1 + score2) / 2
                            if score > best_score:
                                best_score = score
                                border_name = node
                                border_side = indx

                if border_name != 0:  # if there is a match
                    self.rotate(border_name, border_side, 3)  # ROTATE AND ADJUST SIDES FOR MATCH
                    closed_list.append(border_name)  # insert into matched list
                    total_list.remove(border_name)  # remove from unmatched list
                    match_count += 1
                else:  # *NO MATCHES BORDER_NAME = 0
                    print("no matches middle")
                    break

            else:
                print("total list error")
                break

        print("done")

        print(f"{closed_list} CLOSED LIST")
        print(f"{total_list} TOTAL LIST")

        for i in self.piece:
            a = np.degrees(self.piece[i]["rotation"])
            print(f"{i} degrees: {a}")

        # fill up answer
        if len(self.piece) == len(closed_list):
            for c in range(puzzle_width):
                completed_puzzle[0, c] = closed_list.pop(0)
            for r in range(puzzle_height)[1:-1]:  # HEIGHT HERE SHOULD ALWAYS BE >=2 OR FAILS
                completed_puzzle[r, -1] = closed_list.pop(0)
            for c in reversed(range(puzzle_width)):
                completed_puzzle[puzzle_height - 1, c] = closed_list.pop(0)
            for r in reversed(range(puzzle_height - 1)):
                if r > 0:
                    completed_puzzle[r, 0] = closed_list.pop(0)
            if len(self.piece) != 4:  # if not a 2x2 #useless if?
                for r in range(puzzle_height)[1:-1]:
                    for c in range(puzzle_width)[1:-1]:
                        completed_puzzle[r, c] = closed_list.pop(0)
        else:
            completed_puzzle = np.array([])

        if completed_puzzle.size == 0:
            start += 1
            iteration += 1
            if iteration <= 4:
                print(f"******* ITERATION {iteration} ********")
                self.labeling()
                return self.solve(start, iteration)
                pass

        print(completed_puzzle)
        return completed_puzzle

    def display(self):
        for c, i in enumerate(self.piece):
            #CORNER DOTS
            for _, k in enumerate(self.piece[i]["corners"]):
               cv2.circle(self.labelimg, (int(k[0]), int(k[1])), 4, (0, 0, 0), -1)

            # BOUNDING RECT
            x, y, w, h = cv2.boundingRect(self.piece[i]["cont"])
            cv2.rectangle(self.image_test, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(self.foreimg_test, (x, y), (x + 9, y + 9), (255, 255, 255), -1)
            cv2.circle(self.image_test, (x, y), 4, (255, 0, 0), -1)

            # ROTATED RECT/CORNERS
            cv2.drawContours(self.image_test, [self.piece[i]["corners"]], 0, (0, 0, 255), 2)
            cv2.circle(self.image_test, (self.piece[i]["corners"][0][0], self.piece[i]["corners"][0][1]), 4,
                       (255, 0, 0), -1)

            # CENTROIDS AND LABEL
            cv2.putText(self.image_test, f"{i}", (int(self.piece[i]["centroid"][0]), int(self.piece[i]["centroid"][1])),
                        cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            cv2.putText(self.foreimg_test, f"{i}",
                        (int(self.piece[i]["centroid"][0]), int(self.piece[i]["centroid"][1])),
                        cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            cv2.putText(self.labelimg, f"{i}",
                        (int(self.piece[i]["centroid"][0]), int(self.piece[i]["centroid"][1])),
                        cv2.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            cv2.circle(self.binary_test, (int(self.piece[i]["centroid"][0]), int(self.piece[i]["centroid"][1])), 5, 0,
                       -1)

            # FIRST CONTOUR
            cv2.circle(self.image_test, (int(self.piece[i]["cont"][0][0][0]), int(self.piece[i]["cont"][0][0][1])), 8,
                       (255, 255, 0), -1)
            # CONVEX HULL
            cv2.drawContours(self.foreimg_test, [self.piece[i]["hull"]], -1, (255, 255, 255), 2)

            # SIDE CONToUorS AND AREAS
            for k in range(4):
                cv2.drawContours(self.foreimg_test, self.piece[i]["sidecont"][k], -1, (0, 255, 255), 3)
                if self.piece[i]["sides"][k][0] == 1:
                    cv2.drawContours(self.image_test, [self.piece[i]["sidecont"][k]], -1, (255, 0, 0), -1)
                elif self.piece[i]["sides"][k][0] == -1:
                    cv2.drawContours(self.image_test, [self.piece[i]["sidecont"][k]], -1, (0, 255, 255), -1)

        cv2.imshow("test image", self.image_test)
        cv2.imshow("foreimg", self.foreimg_test)
        cv2.imshow("labelimg", self.labelimg)
        cv2.imshow("binarytest", self.binary_test)
        if self.answer.size != 0:
            cv2.imshow("answer", self.answer)
        cv2.waitKey(0)

    def rotate(self, name, matched_side, ref_dir):  # label to be rotated, index, ref_dir of reference piece side
        # align the matched_side to the ref_direction ref_dir
        # ref_dirECTION KEY:
        # 3 = face left 1
        # 2 = face up 0
        # 1 = face right 3
        # 0 = face down 2
        # np.roll + = down ccw//- = up cw
        if 0 > matched_side or matched_side > 3 or 0 > ref_dir or ref_dir > 3:
            print("incorrect parameters to rotate function")
            return

        count = 0
        if ref_dir != 1:
            adjust = m.fabs(ref_dir - 2)
        else:
            adjust = 3
        r = adjust - matched_side

        if 3 > r >= 0 or r == -3:  # ccw
            if r == -3: r = 1
            for t in range(int(m.fabs(r))):
                self.piece[name]["sides"] = np.roll(self.piece[name]["sides"], 1, axis=0)
                self.piece[name]["sidecont"] = np.roll(self.piece[name]["sidecont"], 1, axis=0)
                count += 1
            self.piece[name]["rotation"] -= count * np.pi / 2
        else:  # cw
            if r == 3: r = 1
            for t in range(int(m.fabs(r))):
                self.piece[name]["sides"] = np.roll(self.piece[name]["sides"], -1, axis=0)
                self.piece[name]["sidecont"] = np.roll(self.piece[name]["sidecont"], -1, axis=0)
                count += 1
            self.piece[name]["rotation"] += count * np.pi / 2

    def totalmetric(self, shape_diff, color_diff):
        shape_diff = m.fabs(shape_diff)
        color_diff = m.fabs(color_diff)
        shape_score = 1 / (1 + (0.001 * shape_diff))
        color_score = 1 / (1 + (0.01 * color_diff))
        score = (shape_score * .20) + (color_score * .80)
        return score * 100

    def colormetric(self, name, matched_side, name2, ref_side):
        # metrics: color on side, average color
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))  # 15,15

        temp_dif = m.fabs(
            self.piece[name]["sidecont"][matched_side].shape[0] - self.piece[name2]["sidecont"][ref_side].shape[0])
        end = int(temp_dif // 2)
        start = int(end + (temp_dif % 2 > 0))
        temp_list = []
        c_score = 0
        # useless if?
        if self.piece[name]["sidecont"][matched_side].shape[0] and self.piece[name2]["sidecont"][ref_side].shape[0]:
            if self.piece[name]["sidecont"][matched_side].shape[0] >= self.piece[name2]["sidecont"][ref_side].shape[0]:
                temp_list = self.piece[name]["sidecont"][matched_side][
                            start:len(self.piece[name]["sidecont"][matched_side]) - end]
                temp_list = np.flip(temp_list, axis=0)
                for k, cont in enumerate(self.piece[name2]["sidecont"][ref_side]):
                    Lr, ar, br = self.getNeighbors(self.foreimg, cont[0][0], cont[0][1], kernel)
                    Lm, am, bm = self.getNeighbors(self.foreimg, temp_list[k][0][0], temp_list[k][0][1], kernel)
                    c_score += m.sqrt((Lr - Lm) ** 2 + (ar - am) ** 2 + (br - bm) ** 2)

            else:
                temp_list = self.piece[name2]["sidecont"][ref_side][
                            start:len(self.piece[name2]["sidecont"][ref_side]) - end]
                temp_list = np.flip(temp_list, axis=0)
                for k, cont in enumerate(self.piece[name]["sidecont"][matched_side]):
                    Lm, am, bm = self.getNeighbors(self.foreimg, cont[0][0], cont[0][1], kernel)
                    Lr, ar, br = self.getNeighbors(self.foreimg, temp_list[k][0][0], temp_list[k][0][1], kernel)
                    c_score += m.sqrt((Lr - Lm) ** 2 + (ar - am) ** 2 + (br - bm) ** 2)

        c_score = c_score / (len(temp_list) + 1e-5)
        return c_score

    def getNeighbors(self, img, x, y, kernel):
        # get neighbors 5x5
        # element wise - multiply by the kernel
        # BUT IF INDICES EXCEED FRAME DIMENSIONS

        w = (kernel.shape[1] - 1) // 2
        h = (kernel.shape[0] - 1) // 2
        top = y - h
        bot = (y + h) + 1
        left = x - w
        right = (x + w) + 1
        section = img[top:bot, left:right]  # should be 5,5,3 shape

        count = 0
        L = 0
        a = 0
        b = 0
        section_shape = (section.shape[0],section.shape[1])
        kernel_shape = (kernel.shape[0],kernel.shape[1])
        if section_shape == kernel_shape:
            section = section[kernel == 1]
            for k in section:
                if (k != [0, 0, 0]).all():  # [0,128,128]
                    L += k[0]
                    a += k[1]
                    b += k[2]
                    count += 1
            L = L / (count + 1e-5)
            a = a / (count + 1e-5)
            b = b / (count + 1e-5)
        else:
            print("move pieces away from frame edge")

        return L, a, b

    def headsnholes(self):

        # convexity points
        # circularity = 4piA/(P^2) = 1
        for i in self.piece:
            edges = np.zeros((4, 3))  # top left bottom right
            side_cont = [np.array([]), np.array([]), np.array([]), np.array([])]  # top left bottom right
            hull = cv2.convexHull(self.piece[i]["cont"], returnPoints=False)
            defects = cv2.convexityDefects(self.piece[i]["cont"], hull)

            # Heads/hole algorithm for piece i

            thresh = np.zeros(defects.shape[0])  # convexity points threshold
            for j in range(defects.shape[0]):
                _, _, _, d = defects[j, 0]
                thresh[j] = d
            thresh = np.max(thresh) * .35
            point = []
            start = []
            end = []
            tolerance = 0.35  # head/boundbox area ratio
            box_area = distance(self.piece[i]["corners"][0], self.piece[i]["corners"][1]) * \
                       distance(self.piece[i]["corners"][1], self.piece[i]["corners"][2])
            for j in range(defects.shape[0]):
                s, e, f, d = defects[j, 0]
                if d > thresh:
                    point.append(f)
                    start.append(s)
                    end.append(e)
                    cv2.circle(self.image_test, tuple(self.piece[i]["cont"][f][0]), 5, [0, 0, 255], -1)

            if len(point) > 0:
                loc = {"top": [], "left": [], "bot": [], "right": [], }
                for k in range(len(point)):

                    side = ""
                    minA = 1e10
                    minD = 1e10
                    for j in range(4):
                        a = []
                        if j == 3:
                            a.append(self.piece[i]["corners"][j])
                            a.append(self.piece[i]["corners"][0])
                            a.append(self.piece[i]["cont"][point[k]][0])
                            seg_dis, segment = pnt2line(self.piece[i]["cont"][point[k]][0], self.piece[i]["corners"][j],
                                                        self.piece[i]["corners"][0])
                        else:
                            a.append(self.piece[i]["corners"][j])
                            a.append(self.piece[i]["corners"][j + 1])
                            a.append(self.piece[i]["cont"][point[k]][0])
                            seg_dis, segment = pnt2line(self.piece[i]["cont"][point[k]][0], self.piece[i]["corners"][j],
                                                        self.piece[i]["corners"][j + 1])

                        a = np.array(a)
                        area = cv2.contourArea(a)

                        if j == 0:
                            b = point[k]
                            # minA = area
                            # minD = seg_dis

                        if area + (area * 0.15) <= minA:  # if area significantly smaller
                            minA = area
                            minD = seg_dis
                            b = point[k]
                            if j == 0:
                                side = "left"
                            elif j == 1:
                                side = "top"
                            elif j == 2:
                                side = "right"
                            else:
                                side = "bot"
                        elif area <= minA + (minA * 0.10):  # if area barely smaller
                            # extra check
                            if seg_dis <= minD:  # if
                                minD = seg_dis
                                minA = area
                                b = point[k]
                                if j == 0:
                                    side = "left"
                                elif j == 1:
                                    side = "top"
                                elif j == 2:
                                    side = "right"
                                else:
                                    side = "bot"

                    loc[side].append(b)  # appends current convex defect to a side

                for j, n in enumerate(loc):  # iterates 4 times
                    if len(loc[n]) > 2:  # 2 points associated with side
                        print("threshhold-headnholes.....")
                        break

                    if len(loc[n]) == 2:
                        edges[j][0] = 1  # head
                        cntlist = np.array([])
                        # loc[n].append(loc[n][0])
                        loc[n].sort()
                        for k in range(len(loc[n][0:-1])):  # USELESS FOR

                            cntlist = self.piece[i]["cont"][loc[n][k]:loc[n][k + 1]]
                            head_area = cv2.contourArea(cntlist)

                            if head_area > box_area * tolerance:
                                cntlist = np.concatenate(
                                    (self.piece[i]["cont"][loc[n][k + 1]:len(self.piece[i]["cont"])],
                                     self.piece[i]["cont"][0:loc[n][k]]), axis=0)
                                head_area = cv2.contourArea(cntlist)

                        # cv2.drawContours(self.image_test, [cntlist], -1, (255, 0, 0), -1)

                        edges[j][1] = head_area
                        edges[j][2] = cv2.arcLength(cntlist, False)
                        # COLOR METRIC GOES HERE
                        side_cont[j] = cntlist

                    elif len(loc[n]) == 1:
                        # Possible bug
                        edges[j][0] = -1  # hole
                        s = start[point.index(loc[n][0])]
                        e = end[point.index(loc[n][0])]

                        if s <= e:
                            cntlist = self.piece[i]["cont"][s:e]
                        else:
                            temp = e
                            e = s
                            s = temp
                            cntlist = self.piece[i]["cont"][s:e]

                        hole_area = cv2.contourArea(cntlist)
                        if hole_area > box_area * tolerance:  # if the first cont is inbetween s and e
                            cntlist = np.concatenate((self.piece[i]["cont"][e:len(self.piece[i]["cont"])],
                                                      self.piece[i]["cont"][0:s]),
                                                     axis=0)
                            hole_area = cv2.contourArea(cntlist)

                        # cv2.drawContours(self.image_test, [cntlist], -1, (0, 255, 255), -1)
                        # cut down
                        thresh_d = 3
                        min_s = s
                        min_e = e
                        for indx in range(int(len(cntlist) / 2)):
                            dis, _ = pnt2line(cntlist[indx][0], self.piece[i]["cont"][s][0],
                                              self.piece[i]["cont"][e][0])
                            # dis,_ = pnt2line(cntlist[indx][0],cntlist[0][0],cntlist[len(cntlist)-1][0])
                            if dis > thresh_d:
                                min_s = indx
                                break
                        for indx in range(int(len(cntlist) / 2)):
                            dis, _ = pnt2line(cntlist[len(cntlist) - 1 - indx][0], self.piece[i]["cont"][s][0],
                                              self.piece[i]["cont"][e][0])
                            if dis > thresh_d:
                                min_e = len(cntlist) - 1 - indx
                                break

                        if min_s < min_e:  # safety if
                            cntlist = cntlist[min_s:min_e]

                        hole_area = cv2.contourArea(cntlist)
                        edges[j][1] = hole_area
                        edges[j][2] = cv2.arcLength(cntlist, False)
                        # cv2.drawContours(self.image_test, [cntlist], -1, (0, 255, 255), -1)

                        # COLOR METRIC GOES HERE
                        side_cont[j] = cntlist

                    else:
                        edges[j][0] = 0
                        edges[j][1] = 0
                        # COLOR METRIC GOES HERE
                        edges[j][2] = 0
                        side_cont[j] = np.array([])
            else:
                edges = np.zeros((4, 3))  # top left bottom right
                side_cont = [np.array([]), np.array([]), np.array([]), np.array([])]

            self.piece[i]["sides"] = edges
            self.piece[i]["sidecont"] = np.array(side_cont, dtype=object)
        # GROUPING
        self.type = {"edge": [], "corner": [], "middle": [], "square": [], "unknown": []}
        for i in self.piece:
            axa = np.ones((self.piece[i]["sides"].shape[0],))
            axa[:] = self.piece[i]["sides"][:, 0]
            if np.all(axa != 0):
                self.type["middle"].append(i)
            elif (axa == 0).sum() == 2:
                self.type["corner"].append(i)
            elif (axa == 0).sum() == 1:
                self.type["edge"].append(i)
            elif (axa == 0).sum() == 4:
                self.type["square"].append(i)
            else:
                self.type["unknown"].append(i)

    def corners(self, i, cont, centroid, reorder = True):  # "corners": np.zeros((4, 2))

        # corner detection
        outline = "cont"
        rho = np.zeros((cont.shape[0],), )
        theta = np.zeros((cont.shape[0],))

        for index, c in enumerate(cont):
            for _, k in enumerate(c):
                rho[index], theta[index] = cart2pol(k[0] - centroid[0],
                                                    k[1] - centroid[1])

        if rho[0] < rho[-1]:
            rho = np.append(rho, 0)
        else:
            rho = np.insert(rho, 0, 0)

        peaks = find_peaks(rho, prominence=10)[0]
        if (peaks[0] < 0):
            peaks[0] = peaks[0] + 1
        if (peaks[-1] >= cont.shape[0]):
            peaks[-1] = peaks[-1] - 1


        if rho[0] < rho[-1]:
            rho = np.delete(rho, len(rho) - 1)
        else:
            rho = np.delete(rho, 0)

        # plt.plot(rho)
        # plt.plot(peaks[0], rho[peaks[0]], "x")
        # plt.plot(peaks, rho[peaks], "x")


        peaks, flag = self.reducepeaks(peaks, theta, rho, i)

        if reorder and cont[peaks[-1]][0][0] < cont[peaks[0]][0][0]:
            peaks.insert(0, peaks.pop(-1))

        corners = np.zeros((len(peaks), 2))  # len should be 4
        for index, j in enumerate(peaks):
            corners[index] = cont[j][0]

        return np.int64(corners), flag


    def reducepeaks(self, peaks, theta, rho, i):
        flag=0
        theta = np.degrees(theta)
        peaks = peaks.tolist()
        peaks.sort(key=lambda x: theta[x])

        a = [(x + 360) for x in theta[peaks] if x < 0]
        b = [x for x in theta[peaks] if x >= 0]
        rho_fix = [x for x in rho[peaks]]
        theta_fix = a+b
        theta_fix.append(theta_fix[0])
        rho_fix_indx = np.where(rho_fix == np.min(rho_fix))[0][0]

        break_num = 0
        while len(peaks) > 4:
            break_num += 1
            errors = []
            for k in range(len(theta_fix))[0:-1]:
                if m.fabs(theta_fix[k] - theta_fix[k+1]) <= 180:
                    temp_val = m.fabs(theta_fix[k] - theta_fix[k+1])
                else:
                    temp_val = 360 - m.fabs(theta_fix[k] - theta_fix[k+1])

                errors.append(m.fabs(temp_val - 90))

            error_thresh = np.max(errors) * .50
            k=0
            while k < len(errors):
                if break_num == 1:
                    k = rho_fix_indx
                    break_num += 1
                if k < len(errors)-1:
                    if errors[k] >= error_thresh and errors[k+1] >= error_thresh:
                        peaks.pop(k+1)
                        theta_fix.pop(k+1)
                        break
                else:
                    if errors[k] >= error_thresh and errors[0] >= error_thresh:
                        peaks.pop(0)
                        theta_fix.pop(0)
                        break
                k+=1

            if break_num > 50:
                print("Corner break error", i)
                flag = 1
                break

        if len(peaks) < 4:
            print("CORNER ERROR<4", i)
            flag = 1

        return peaks, flag

    def labeling(self):

        contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i, cont in enumerate(contours):
            hull = cv2.convexHull(cont)

            M = cv2.moments(cont)
            center = (M['m10'] / M['m00'], M['m01'] / M['m00'])

            # bounding rectangle
            # x, y, w, h = cv2.boundingRect(cont)
            # cv2.rectangle(self.image_test, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.circle(self.image_test, (x, y), 4, (255, 0, 0), -1)

            # rotated rect
            rect = cv2.minAreaRect(cont)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            self.piece[i + 1] = {"cont": cont, "hull": hull, "centroid": center, "corners": box, "rotation": 0.,
                                 "sides": np.zeros((4, 3)),
                                 "sidecont": np.array([np.array([]), np.array([]), np.array([]), np.array([])],
                                               dtype=object)}
            corner, flag = self.corners(i+1, cont,center)
            if not flag:
                self.piece[i+1]["corners"] = corner
            else:
                self.cornerflag = 1

        self.orientation()
        self.headsnholes()

    def orientation(self):
        for i in self.piece:
            # cv2.drawContours(self.image_test, [self.piece[i]["corners"]], 0, (0, 0, 255), 2)
            # cv2.circle(self.image_test, (self.piece[i]["corners"][3][0], self.piece[i]["corners"][3][1]), 4, (255, 0, 0), -1)
            # cv2.line(self.image_test, self.piece[i]["corners"][1], self.piece[i]["corners"][2], [0, 255, 0], 2)
            _, theta = cart2pol(self.piece[i]["corners"][2][0] - self.piece[i]["corners"][1][0],
                                self.piece[i]["corners"][2][1] - self.piece[i]["corners"][1][1])
            self.piece[i]["rotation"] = -theta

    def show_solved(self):
        solvedimg = np.zeros(self.solution_matrix.shape, dtype=object)

        for r, rows in enumerate(self.solution_matrix):
            for c, i in enumerate(rows):
                width = int(edistance(self.piece[i]["corners"][1], self.piece[i]["corners"][2]))
                height = int(edistance(self.piece[i]["corners"][1], self.piece[i]["corners"][0]))
                pt1 = []
                pt1.append(self.piece[i]["corners"][1])
                pt1.append(self.piece[i]["corners"][2])
                pt1.append(self.piece[i]["corners"][0])
                pt1.append(self.piece[i]["corners"][3])
                pt1 = np.float32(pt1)
                pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                matrix = cv2.getPerspectiveTransform(pt1, pt2)
                img = cv2.warpPerspective(self.labelimg, matrix, (width, height))

                _, theta = cart2pol(self.piece[i]["corners"][2][0] - self.piece[i]["corners"][1][0],
                                   self.piece[i]["corners"][2][1] - self.piece[i]["corners"][1][1])
                theta = -theta
                theta = self.piece[i]["rotation"] - theta if self.piece[i]["rotation"] < 0 else self.piece[i][
                                                                                                    "rotation"] + theta
                img = rotate_image(img, -np.degrees(theta))
                solvedimg[r,c] = img
                #cv2.imshow("solution", img)
                #cv2.waitKey(0)

        WOW = stackImages(1, solvedimg.tolist())
        return WOW

    def placeheadnhole(self,cflag):
        if cflag:
            return None,1
        section = np.zeros((self.binary.shape[0], self.binary.shape[1], 3), dtype=np.uint8)
        kernel_dia = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        r_start = 0
        c_start = 0
        for r, rows in enumerate(self.solution_matrix):
            for c, i in enumerate(rows):

                x, y, w, h = cv2.boundingRect(self.piece[i]["cont"])
                crop_bin = self.binary[y:y+h,x:x+w]
                img = self.foreground[y:y+h,x:x+w]


                img = rotate_image(img, -np.degrees(self.piece[i]["rotation"]))
                crop_bin = rotate_image(crop_bin, -np.degrees(self.piece[i]["rotation"]))

                crop_bin = self.Seg.removenoise(crop_bin)
                img = cv2.bitwise_and(img,img,mask=crop_bin)


                contours, _ = cv2.findContours(crop_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if len(contours) != 1:
                    print("display error", i)
                    return None,1
                x, y, w, h = cv2.boundingRect(contours[0])
                reduced_img = img[y:y + h, x:x + w]
                reduced_bin = crop_bin[y:y + h, x:x + w]
                reduced_bin = cv2.morphologyEx(reduced_bin, cv2.MORPH_OPEN, kernel_dia, iterations=1)

                contours1, _ = cv2.findContours(reduced_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                M = cv2.moments(contours1[0])
                center = (M['m10'] / M['m00'], M['m01'] / M['m00'])
                subcorner,flag = self.corners(i,contours1[0],center, False)
                if flag:
                    # print("corner display error", i)
                    # for c in subcorner:
                    #     cv2.circle(reduced_img,(c[0],c[1]),10,0,-1)
                    # cv2.imshow("sdf",reduced_img)
                    # cv2.imshow("sdgfs",reduced_bin)
                    # cv2.waitKey(0)
                    return None, 1
                else:

                    rows, cols, _ = reduced_img.shape

                    #VERTICAL
                    if subcorner[0][1] > subcorner[1][1]:
                        black_r_tgap = subcorner[1][1]
                    else:
                        black_r_tgap = subcorner[0][1]

                    if subcorner[2][1] > subcorner[3][1]:
                        black_r_bgap = subcorner[2][1]
                    else:
                        black_r_bgap = subcorner[3][1]
                    if r_start ==0:
                        r_display_start = r_start
                    else:
                        r_display_start = int(m.fabs(r_start - black_r_tgap))
                    #HORIZONTAL
                    if subcorner[0][0] > subcorner[3][0]:
                        black_c_lgap = subcorner[3][0]
                    else:
                        black_c_lgap = subcorner[0][0]
                    if subcorner[1][0] > subcorner[2][0]:
                        black_c_rgap = subcorner[1][0]
                    else:
                        black_c_rgap = subcorner[2][0]
                    if c_start ==0:
                        c_display_start = c_start
                    else:
                        c_display_start = int(m.fabs(c_start - black_c_lgap))


                    roi = section[r_display_start:r_display_start+rows, c_display_start:c_display_start+cols]
                    display = cv2.bitwise_or(reduced_img,roi)
                    section[r_display_start:r_display_start+rows,c_display_start:c_display_start+cols] = display
                    c_start += black_c_rgap - black_c_lgap
                #cv2.imshow("c", reduced_bin)
                #cv2.waitKey(0)
            r_start += black_r_bgap- black_r_tgap
            if i != self.solution_matrix[-1,-1]:
                c_start = 0

        #REDUCE FRAME SIZE
        section = section[:r_start,:c_start]

        return section, 0


if __name__ == '__main__':
    puzzle = PuzzleSolver(LOCATION, 1)

    cv2.waitKey(0)
    pass
