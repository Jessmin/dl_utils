import re
from hanziconv import HanziConv
import editdistance
import numpy as np
import Polygon as plg


def gen_input(file_path: str, pred=False):
    pointsList = []
    confidenceList = []
    transcriptionsList = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            line_arr = line.split(',')
            points = line_arr[0:8]
            if pred:
                confidence = float(line_arr[8])
                confidenceList.append(confidence)
                transcriptions = ''.join(line_arr[9:])
            else:
                transcriptions = ''.join(line_arr[8:])
            pointsList.append(points)
            transcriptionsList.append(transcriptions)

    result = (pointsList, confidenceList, transcriptionsList)
    return result


def polygon_from_points(points, correctOffset=False):
    """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """
    if correctOffset:  # this will substract 1 from the coordinates that correspond to the xmax and ymax
        points[2] -= 1
        points[4] -= 1
        points[5] -= 1
        points[7] -= 1

    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)


def get_intersection_over_union(pD, pG):
    try:
        return get_intersection(pD, pG) / get_union(pD, pG)
    except Exception:
        return 0


def transcription_match(transGt,
                        transDet,
                        specialCharacters='!?.:,*"()·[]/\'',
                        onlyRemoveFirstLastCharacterGT=True):

    if onlyRemoveFirstLastCharacterGT:
        # special characters in GT are allowed only at initial or final position
        if (transGt == transDet):
            return True

        if specialCharacters.find(transGt[0]) > -1:
            if transGt[1:] == transDet:
                return True

        if specialCharacters.find(transGt[-1]) > -1:
            if transGt[0:len(transGt) - 1] == transDet:
                return True

        if specialCharacters.find(transGt[0]) > -1 and specialCharacters.find(
                transGt[-1]) > -1:
            if transGt[1:len(transGt) - 1] == transDet:
                return True
        return False
    else:
        # Special characters are removed from the begining and the end of both Detection and GroundTruth
        while len(transGt) > 0 and specialCharacters.find(transGt[0]) > -1:
            transGt = transGt[1:]

        while len(transDet) > 0 and specialCharacters.find(transDet[0]) > -1:
            transDet = transDet[1:]

        while len(transGt) > 0 and specialCharacters.find(transGt[-1]) > -1:
            transGt = transGt[0:len(transGt) - 1]

        while len(transDet) > 0 and specialCharacters.find(transDet[-1]) > -1:
            transDet = transDet[0:len(transDet) - 1]

        return transGt == transDet


# #from RTWC17
def normalize_txt(st):
    """
    Normalize Chinese text strings by:
        - remove puncutations and other symbols
        - convert traditional Chinese to simplified
        - convert English characters to lower cases
    """
    st = ''.join(st.split(' '))
    st = re.sub("\"", "", st)
    # remove any this not one of Chinese character, ascii 0-9, and ascii a-z and A-Z
    new_st = re.sub(r'[^\u4e00-\u9fa5\u0041-\u005a\u0061-\u007a0-9]+', '', st)
    # convert Traditional Chinese to Simplified Chinese
    new_st = HanziConv.toSimplified(new_st)
    # convert uppercase English letters to lowercase
    new_st = new_st.lower()
    return new_st


def text_distance(str1, str2):
    str1 = normalize_txt(str1)
    str2 = normalize_txt(str2)
    return editdistance.eval(str1, str2)


def compute_ap(confList, matchList, numGtCare):
    correct = 0
    AP = 0
    if len(confList) > 0:
        confList = np.array(confList)
        matchList = np.array(matchList)
        sorted_ind = np.argsort(-confList)
        confList = confList[sorted_ind]
        matchList = matchList[sorted_ind]
        for n in range(len(confList)):
            match = matchList[n]
            if match:
                correct += 1
                AP += float(correct) / (n + 1)

        if numGtCare > 0:
            AP /= numGtCare

    return AP
