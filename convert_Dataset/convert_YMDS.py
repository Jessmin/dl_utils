#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@name    :convert_YMDS.py
@statement    :把lableme标注的四点数据转换为正视图的四点标注
@time    :2021/07/07 17:01:22
@author    :zhaohj
'''
import numpy as np
import cv2
import imutils
import math
import glob
import os
import json


def get_min_square(label_data):
    """生成最小外接矩形

    Args:
        label_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    shapes = label_data['shapes']
    boxes = []
    for shape in shapes:
        points = shape['points']
        cnt = np.asarray(points, 'float32')
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
    return boxes


def rotate_points(points, M):
    pts = np.float32(points).reshape([-1, 2])
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(M, pts)
    target_point = [[target_point[0][x], target_point[1][x]]
                    for x in range(len(target_point[0]))]
    return np.int0(target_point)


def rotate_tables(img, boxes):
    """根据内部表格矩形框旋转图片
    1.若有多个表格，只根据第一个做旋转，后续的表格调整其最大外接矩阵为正矩阵
    """
    points_list = []
    for idx, box in enumerate(boxes):
        if idx == 0:
            p1, p2, p3, p4 = box
            theta =  (p2[0] - p1[0])/(p2[1] - p1[1]) 
            angle = math.atan(theta)
            h, w, c = img.shape
            center = (w // 2, h // 2)
            cv2.polylines(img,[np.int32(box)], True,(0,255,255),4)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # img = cv2.warpAffine(
            #     img,
            #     M, (w, h),
            #     flags=cv2.INTER_CUBIC,
            #     borderMode=cv2.BORDER_REPLICATE)
            # points = rotate_points(box, M)
            points = box
            points_list.append(points)
        else:
            pass
    return img, points_list


def run():
    img_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/images'
    label_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableDetection'
    xml_files = glob.glob(f'{label_dir}/*.json')
    print(len(xml_files))
    for xml_file in xml_files:
        _, filename = os.path.split(xml_file)
        filename, _ = os.path.splitext(filename)
        img_path = os.path.join(img_dir, f'{filename}.png')
        img = cv2.imread(img_path)
        with open(xml_file, 'r') as f:
            data = json.load(f)
            boxes = get_min_square(data)
            img, points_list = rotate_tables(img, boxes)
            for points in points_list:
                pt1 = points[0]
                pt2 = points[2]
                points = np.int32(points).reshape((-1, 1, 2))
                # cv2.rectangle(img, pt1, pt2, (255, 255, 255), -1)
                # cv2.fillPoly(img, [points],(255, 0, 255))
            img_show = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
            cv2.imshow("img", img_show)
            cv2.waitKey(0)


if __name__ == '__main__':
    run()
