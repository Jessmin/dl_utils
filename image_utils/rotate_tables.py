from paddleocr import PaddleOCR
import numpy as np
import cv2
import math
import os
from tqdm import tqdm
import glob

"""
使用文字检测框的结果对图片进行矫正
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


def predict_det(img):
    box_data = ocr.ocr(img, rec=False)
    return box_data


def inpaint_with_ocr(bboxes, img: np.ndarray):
    ocr_img = img.copy()
    for box in bboxes:
        cv2.fillPoly(ocr_img, [np.array(box, np.int32)], (255, 255, 255))
    return ocr_img


def get_rotation_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)  # canny， 便于hough line减少运算量
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=60)  # 参数很关键
    lengths = []  # 存储所有线的坐标、长度
    if lines is None:
        return 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5  # 勾股定理，求直线长度
        angle = math.atan((y2 - y1) / (x2 - x1))
        angle = angle * (180 / math.pi)
        lengths.append([x1, y1, x2, y2, angle, length])
    # 绘制最长的直线
    lengths.sort(key=lambda x: x[-1])
    longest_line = lengths[-1]
    x1, y1, x2, y2, angle, length = longest_line
    return angle


def rotate_bound(image, angle):
    # 旋转中心点，默认为图像中心点
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # 得到旋转矩阵，1.0表示与原图大小一致
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 计算旋转后的图像大小（避免图像裁剪）
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # 调整旋转矩阵（避免图像裁剪）
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # 执行仿射变换、得到图像
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


def run(src_img):
    box_data = predict_det(src_img)
    ocr_img = inpaint_with_ocr(box_data, src_img)
    ret, dst = cv2.threshold(ocr_img, 150, 230, cv2.THRESH_TOZERO_INV)
    dst = 255 - dst
    angle = get_rotation_angle(dst)
    imag = rotate_bound(src_img, angle)
    return imag


def get_rotate_imgs():
    src_img_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation'
    output_img_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation_rotated'
    os.makedirs(output_img_dir, exist_ok=True)
    imgs_list = glob.glob(f'{src_img_dir}/*.png')
    for i in tqdm(range(len(imgs_list))):
        src_img_path = imgs_list[i]
        _, filename = os.path.split(src_img_path)
        output_filepath = os.path.join(output_img_dir, filename)
        src_img = cv2.imread(src_img_path)
        imag = run(src_img)
        cv2.imwrite(output_filepath, imag)
        break


if __name__ == '__main__':
    get_rotate_imgs()
