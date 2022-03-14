from cv2 import cv2
import math as m

def distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    dis1 = m.fabs(x2 - x1)
    dis2 = m.fabs(y2 - y1)
    # if dis1 >= dis2:
    #     return dis1
    # else:
    #     return dis2
    return m.sqrt(dis1**2 + dis2**2)

def position(piecedict ,piece1, piece2, type):

    #piecedict = piece dictionary
    #type = 0 horizontal distance, piece1 is left piece, piece2 is right piece
    #type = 1 vertical distance, piece1 is top piece, piece2 is bottom piece

    max_val = 0
    min_val = 1e10

    if not type:
        if piecedict[piece1]["sides"][3][0] == 1 and piecedict[piece2]["sides"][1][0] == -1:

            for k in piecedict[piece1]["sidecont"][3]:
                dis = distance(piecedict[piece1]["centroid"], k[0])
                if dis > max_val:
                    max_val = dis

            for k in piecedict[piece2]["sidecont"][1]:
                dis = distance(piecedict[piece2]["centroid"], k[0])
                if dis < min_val:
                    min_val = dis

            return min_val + max_val


        elif piecedict[piece1]["sides"][3][0] == -1 and piecedict[piece2]["sides"][1][0] == 1:

            for k in piecedict[piece1]["sidecont"][3]:
                dis = distance(piecedict[piece1]["centroid"], k[0])
                if dis < min_val:
                    min_val = dis

            for k in piecedict[piece2]["sidecont"][1]:
                dis = distance(piecedict[piece2]["centroid"], k[0])
                if dis > max_val:
                    max_val = dis


            return min_val + max_val
        else:
            print("horizontal error mismatch")
            return 0
    else:
        if piecedict[piece1]["sides"][2][0] == 1 and piecedict[piece2]["sides"][0][0] == -1:

            for k in piecedict[piece1]["sidecont"][2]:
                dis = distance(piecedict[piece1]["centroid"], k[0])
                if dis > max_val:
                    max_val = dis
                    #a = k
            for k in piecedict[piece2]["sidecont"][0]:
                dis = distance(piecedict[piece2]["centroid"], k[0])
                if dis < min_val:
                    min_val = dis
                    #b = k
            # print(max_val, a)
            # print(min_val, b)
            return min_val + max_val

        elif piecedict[piece1]["sides"][2][0] == -1 and piecedict[piece2]["sides"][0][0] == 1:

            for k in piecedict[piece1]["sidecont"][2]:
                dis = distance(piecedict[piece1]["centroid"], k[0])
                if dis < min_val:
                    min_val = dis

            for k in piecedict[piece2]["sidecont"][0]:
                dis = distance(piecedict[piece2]["centroid"], k[0])
                if dis > max_val:
                    max_val = dis

            return min_val + max_val
        else:
            print("vertical error mismatch")
            return 0

if __name__ == '__main__':
    a = [1,2,3,4]
    a = [i for i in a if i>2]
    print(a)
    pass