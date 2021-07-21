import numpy as np
from utils import polygon_from_points, get_intersection, get_intersection_over_union, transcription_match, text_distance, gen_input, compute_ap
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', dest='gt_dir', type=str, required=True)
    parser.add_argument('--subm', dest='subm_dir', type=str, required=True)
    args = parser.parse_args()
    return args


def evaluate_method(gt, subm, arrGlobalConfidences, arrGlobalMatches):
    recall = 0
    precision = 0
    hmean = 0
    detCorrect = 0
    iouMat = np.empty([1, 1])
    gtPols = []
    detPols = []
    gtTrans = []
    detTrans = []
    gtPolPoints = []
    detPolPoints = []
    gtDontCarePolsNum = []
    detDontCarePolsNum = []
    detMatchedNums = []
    sampleAP = 0
    pointsList, _, transcriptionsList = gt
    for n in range(len(pointsList)):
        points = pointsList[n]
        transcription = transcriptionsList[n]
        dontCare = transcription == "###"
        gtPol = polygon_from_points(points)
        gtPols.append(gtPol)
        gtPolPoints.append(points)
        gtTrans.append(transcription)
        if dontCare:
            gtDontCarePolsNum.append(len(gtPols) - 1)
    # end
    # compare with subm
    pointsList, confidencesList, transcriptionsList = subm
    for n in range(len(pointsList)):
        points = pointsList[n]
        transcription = transcriptionsList[n]
        detPol = polygon_from_points(points)
        detPols.append(detPol)
        detPolPoints.append(points)
        detTrans.append(transcription)
        if len(gtDontCarePolsNum) > 0:
            for dontCarePol in gtDontCarePolsNum:
                dontCarePol = gtPols[dontCarePol]
                intersected_area = get_intersection(dontCarePol, detPol)
                pdDimensions = detPol.area()
                precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                if precision > 0.5:
                    detDontCarePolsNum.append(len(detPols) - 1)
                    break
    # endqq
    if len(gtPols) > 0 and len(detPols) > 0:
        # Calculate IoU and precision matrixs
        outputShape = [len(gtPols), len(detPols)]
        iouMat = np.empty(outputShape)
        gtRectMat = np.zeros(len(gtPols), np.int8)
        detRectMat = np.zeros(len(detPols), np.int8)
        for gtNum in range(len(gtPols)):
            for detNum in range(len(detPols)):
                pG = gtPols[gtNum]
                pD = detPols[detNum]
                iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
        # end
    iou_threshold = 0.5
    detCorrect = 0
    txtCorrect = 0
    iouCorrect = 0
    for gtNum in range(len(gtPols)):
        for detNum in range(len(detPols)):
            if gtRectMat[gtNum] == 0 and detRectMat[
                    detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                if iouMat[gtNum, detNum] > iou_threshold:
                    iouCorrect += 1
                    gtRectMat[gtNum] = 1
                    detRectMat[detNum] = 1
                    correct = transcription_match(gtTrans[gtNum].upper(),
                                                  detTrans[detNum].upper())
                    txt_distance = text_distance(gtTrans[gtNum].upper(),
                                                 detTrans[detNum].upper())
                    if txt_distance < len(gtTrans[gtNum].upper()) / 5:
                        txtCorrect += 1
                    if correct:
                        detCorrect += 1
                        detMatchedNums.append(detNum)
    # end
    for detNum in range(len(detPols)):
        if detNum not in detDontCarePolsNum:
            match = detNum in detMatchedNums
            arrGlobalConfidences.append(confidencesList[detNum])
            arrGlobalMatches.append(match)
    # end
    numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
    numDetCare = (len(detPols) - len(detDontCarePolsNum))
    if numGtCare == 0:
        recall = float(1)
        precision = float(0) if numDetCare > 0 else float(1)
        sampleAP = precision
    else:
        recall = float(detCorrect) / numGtCare
        precision = 0 if numDetCare == 0 else float(detCorrect) / numDetCare
    hmean = 0 if (precision + recall
                  ) == 0 else 2.0 * precision * recall / (precision + recall)
    metrics = {
        'precision': precision,
        'recall': recall,
        'hmean': hmean,
        'detCorrect': detCorrect,
        'iouCorrect': iouCorrect,
        'txtCorrect': txtCorrect,
        'numGtCare': numGtCare,
        'numDetCare': numDetCare,
        'AP': sampleAP,
    }
    return metrics


def evaluate_dir(gt_dir, pred_dir):
    matchedSum = 0
    iouMatchedSum = 0
    txtMatchedSum = 0
    numGlobalCareGt = 0
    numGlobalCareDet = 0
    arrGlobalConfidences = []
    arrGlobalMatches = []
    filelist = os.listdir(gt_dir)
    for filename in filelist:
        gt_filename = os.path.join(gt_dir, filename)
        pred_filename = os.path.join(pred_dir, filename)
        if os.path.exists(pred_filename):
            gt = gen_input(gt_filename)
            pred = gen_input(pred_filename, pred=True)
            metics = evaluate_method(gt, pred, arrGlobalConfidences,
                                     arrGlobalMatches)
            detCorrect = metics['detCorrect']
            iouCorrect = metics['iouCorrect']
            txtCorrect = metics['txtCorrect']
            numGtCare = metics['numGtCare']
            numDetCare = metics['numDetCare']
            matchedSum += detCorrect
            iouMatchedSum += iouCorrect
            txtMatchedSum += txtCorrect
            numGlobalCareGt += numGtCare
            numGlobalCareDet += numDetCare
    methodRecall = 0 if numGlobalCareGt == 0 else float(
        matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(
        matchedSum) / numGlobalCareDet
    methodIouPrecision = 0 if numGlobalCareDet == 0 else float(
        iouMatchedSum) / numGlobalCareDet
    methodTxtPrecision = 0 if numGlobalCareDet == 0 else float(
        txtMatchedSum) / numGlobalCareDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
        methodRecall + methodPrecision)
    AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)
    methodMetrics = {
        'end-to-end-precision': methodPrecision,
        'iou-precision': methodIouPrecision,
        'txt-precision': methodTxtPrecision,
        'recall': methodRecall,
        'hmean': methodHmean,
        'AP': AP
    }
    print(methodMetrics)
    return methodMetrics


if __name__ == '__main__':
    args = parse_args()
    evaluate_dir(args.gt_dir, args.subm_dir)
