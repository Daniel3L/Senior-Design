from cv2 import cv2
import numpy as np
from imageprocessing import Segmentation
import math as m


def deltaE(E, L, C1, C2):
    K1 = .045
    K2 = 0.015
    Sc = 1 + K1 * C1
    Sh = 1 + K2 * C1
    C = C1 - C2
    H = m.sqrt(E ** 2 - L ** 2 - C ** 2)
    dE = m.sqrt(L ** 2 + (C / Sc) ** 2 + (H / Sh) ** 2)
    return dE

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
    def __init__(self, puzzle_piece, flag=0):
        self.Seg = Segmentation()

        self.orig = puzzle_piece
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
                self.answer = self.show_solved()
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
                        t56 = 1
                        if t5 and t5 != -t6:
                            t56 = 0
                        if t1 == -t2 and t3 == -t4 and t56:  # potential pair if heads and hole
                            # METRIC GOES HERE FINDING BEST MATCH
                            overlap_diff = m.fabs(a1 - a2)
                            overlap_diff2 = m.fabs(a3 - a4)
                            color_diff = self.colormetric(node, indx, last_matched, 3)
                            color_diff2 = self.colormetric(node, side_location, top_matched, 2)
                            score1 = self.totalmetric(overlap_diff, color_diff)
                            score2 = self.totalmetric(overlap_diff2, color_diff2)
                            score = (score1 + score2) / 2
                            if score1 > best_score:
                                best_score = score1
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
            # for _, k in enumerate(self.piece[c]['corners']):
            #    cv2.circle(self.image_test, (int(k[0]), int(k[1])), 4, (0, 0, 0), -1)
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # 15,15

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
        section = section[kernel == 1]

        count = 0
        L = 0
        a = 0
        b = 0
        for k in section:
            if (k != [0, 0, 0]).all():  # [0,128,128]
                L += k[0]
                a += k[1]
                b += k[2]
                count += 1
        L = L / (count + 1e-5)
        a = a / (count + 1e-5)
        b = b / (count + 1e-5)

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

                if m.fabs(np.degrees(self.piece[i]["rotation"])) >= 360:
                    print("degree error piece: ",i)
                elif m.fabs(np.degrees(self.piece[i]["rotation"])) >= 270:
                    if self.piece[i]["rotation"] >= 0:
                        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        print("error rotation piece: ",i)
                    else:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif m.fabs(np.degrees(self.piece[i]["rotation"])) >= 180:
                    #img = cv2.rotate(img, cv2.ROTATE_180)
                    if self.piece[i]["rotation"] >= 0:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                elif m.fabs(np.degrees(self.piece[i]["rotation"])) >= 90:
                    if self.piece[i]["rotation"] >= 0:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    else:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    if self.piece[i]["rotation"] >= 0:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                solvedimg[r,c] = img
                #cv2.imshow("solution", img)
                #cv2.waitKey(0)

        WOW = stackImages(1, solvedimg.tolist())
        return WOW


if __name__ == '__main__':
    location = "C:/Users/danie/PycharmProjects/pythonProject/images"
    puzzle = PuzzleSolver(cv2.imread(location + f"/piece{0}.png"), 1)

    cv2.waitKey(0)
    pass
