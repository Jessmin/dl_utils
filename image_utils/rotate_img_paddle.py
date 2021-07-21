from paddleocr import PaddleOCR
import numpy as np
import cv2
import math
import os
from tqdm import tqdm
import glob
from scipy.stats import mode
from math import fabs, cos, sin, radians

"""
使用paddle 文字方向分类器对图片文字方向做分类
"""
ocr = PaddleOCR(
    det_model_dir=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/det',
    rec_model_dir=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/rec',
    rec_char_dict_path=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/ppocr_keys_v1.txt',
    cls_model_dir=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/cls',
    use_angle_cls=True,
    max_text_length=15,
    drop_score=0.6,
    det_db_unclip_ratio=2.0,
    lang="ch")


def predict_direction(img):
    result = ocr.ocr(img, det=False, rec=False, cls=True)
    return int(result[0][0])


def rotate_image(image, degree):
    if -45 <= degree <= 0:
        degree = degree  # 负角度 顺时针
    if -90 <= degree < -45:
        degree = 90 + degree  # 正角度 逆时针
    if 0 < degree <= 45:
        degree = degree  # 正角度 逆时针
    if 45 < degree < 90:
        degree = degree - 90  # 负角度 顺时针
    # print("rotate degree:", degree)
    # # 获取旋转后4角的填充色
    filled_color = -1
    if filled_color == -1:
        filled_color = mode(
            [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]).mode[0]
    if np.array(filled_color).shape[0] == 2:
        if isinstance(filled_color, int):
            filled_color = (filled_color, filled_color, filled_color)
    else:
        filled_color = tuple([int(i) for i in filled_color])

    height, width = image.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree,
                                          1)  # 逆时针旋转 degree

    matRotation[0, 2] += (
                                 widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(
        image, matRotation, (widthNew, heightNew), borderValue=filled_color)
    return imgRotation


if __name__ == '__main__':
    img_path = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation/0017-1.png'
    img = cv2.imread(img_path)
    result = predict_direction(img)
    rotated_img = rotate_image(img,result)
    result = predict_direction(rotated_img)
    print(result)