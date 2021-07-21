import cv2
import numpy as np
import math


# from cocoNLP.extractor import extractor
import cv2
import numpy as np
import os
import time
import glob
from tqdm import tqdm


def predict(img, ocr):
    ocr_data = ocr.ocr(img, cls=True)
    return ocr_data


def rotate_bound(image, angle):
    # 旋转中心点，默认为图像中心点
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # 给定旋转角度后，得到旋转矩阵
    # 数学原理：
    #       https://blog.csdn.net/liyuan02/article/details/6750828
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)  # 得到旋转矩阵，1.0表示与原图大小一致
    # print("RotationMatrix2D：\n", M)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算旋转后的图像大小（避免图像裁剪）
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵（避免图像裁剪）
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # print("RotationMatrix2D：\n", M)

    # 执行仿射变换、得到图像
    return cv2.warpAffine(
        image, M, (nW, nH),
        # borderMode=cv2.BORDER_DEFAULT
        borderValue=(255, 255, 255)
    )  # BORDER_REPLICATE)
    # borderMode=cv2.BORDER_REPLICATE 使用边缘值填充
    # 或使用borderValue=(255,255,255)) # 使用常数填充边界（0,0,0）表示黑色


def get_points(coordinate):
    center = coordinate[0]
    for _ in range(1, 4):
        center = center + coordinate[_]
    center = center / 4
    coordinate_temp = coordinate.copy()  # 复制一份坐标，避免原坐标被破坏
    left_coordinate = []  # 存储x轴小于中心坐标点的点
    delete_index = []
    left_top, right_top, right_bottom, left_bottom = 0, 0, 0, 0
    # 将 x轴小于中心坐标点的点 存储进left_coordinate
    for _ in range(4):
        if (coordinate[_][0] < center[0]):
            left_coordinate.append(coordinate[_])
            delete_index.append(_)
    # 将存储进 left_coordinate 的元素从coordinate_temp中删除
    coordinate_temp = np.delete(coordinate_temp, delete_index, axis=0)
    left_coordinate_temp = left_coordinate.copy()  # 避免程序过程因为left_coordinate的变动而导致最初的条件判断错误

    if len(left_coordinate_temp) == 2:
        # 比较左边两个点的y轴，大的为左上
        if left_coordinate[0][1] < left_coordinate[1][1]:
            left_bottom = left_coordinate[0]
            left_top = left_coordinate[1]
        elif left_coordinate[0][1] > left_coordinate[1][1]:
            left_bottom = left_coordinate[1]
            left_top = left_coordinate[0]
        # 比较右边两个点的y轴，大的为右上
        if coordinate_temp[0][1] < coordinate_temp[1][1]:
            right_bottom = coordinate_temp[0]
            right_top = coordinate_temp[1]
        elif coordinate_temp[0][1] > coordinate_temp[1][1]:
            right_bottom = coordinate_temp[1]
            right_top = coordinate_temp[0]
    elif (len(left_coordinate_temp) == 1):
        left_bottom = left_coordinate[0]
        delete_index = []
        for _ in range(3):
            if (coordinate_temp[_][0] == center[0] and coordinate_temp[_][1] > center[1]):
                left_top = coordinate_temp[_]
                delete_index.append(_)
            if (coordinate_temp[_][0] == center[0] and coordinate_temp[_][1] < center[1]):
                right_bottom = coordinate_temp[_]
                delete_index.append(_)
        coordinate_temp = np.delete(coordinate_temp, delete_index, axis=0)
        right_top = coordinate_temp[0]
    return left_top, right_top, right_bottom, left_bottom


def get_rotate_angle_with_boxes(boxes, img):
    lines = []
    out_boxes = []
    for box in boxes:
        p1, p2, p3, p4 = get_points(box)
        if len(np.array(p1).shape) == 0 or len(np.array(p2).shape) == 0:
            continue
        x1, y1 = p1
        x2, y2 = p2
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        angle = math.atan((y2 - y1) / (x2 - x1))
        angle = angle * (180 / math.pi)
        lines.append([x1, y1, x2, y2, angle, length])
        out_boxes.append([box, angle, length])
    lines.sort(key=lambda x: x[-1])
    if len(lines)>6:
        angles = [x[-2] for x in lines[-5:-1]]
        angle = np.mean(angles)
    else:
        angle=0
    #
    out_boxes.sort(key=lambda x: x[-1])
    filtered_boxes = out_boxes[-5:-1]
    for b in filtered_boxes:
        box, _, _ = b
        cv2.fillPoly(img,[np.array(box,np.int32)], (255,255,255))
    # cv2.imshow('im', img)
    # cv2.waitKey(0)
    # longest_line = lines[-1]
    # x1, y1, x2, y2, length = longest_line
    # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 4)  # 绘制直线（红色）
    #
    # if x2 == x1:
    #     angle = 0
    # else:
    #     angle = math.atan((y2 - y1) / (x2 - x1))
    #     # print("angle-radin:", angle)  # 弧度形式
    #     angle = angle * (180 / math.pi)
    #     # print("angle-degree:", angle)  # 角度形式
    return angle


def get_rotation_angle(image, show_longest_line=True, show_all_lines=False):
    # image = image.copy()  # 复制备份，因为line（）函数为in-place
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)  # canny， 便于hough line减少运算量
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=60)  # 参数很关键
    # minLineLengh(线的最短长度，比这个短的都被忽略)
    # maxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）
    # 函数cv2.HoughLinesP()是一种概率直线检测，原理上讲hough变换是一个耗时耗力的算法，
    # 尤其是每一个点计算，即使经过了canny转换了有的时候点的个数依然是庞大的，
    # 这个时候我们采取一种概率挑选机制，不是所有的点都计算，而是随机的选取一些个点来计算，相当于降采样。
    lengths = []  # 存储所有线的坐标、长度
    if lines is None:
        return 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5  # 勾股定理，求直线长度
        angle = math.atan((y2 - y1) / (x2 - x1))
        angle = angle * (180 / math.pi)
        lengths.append([x1, y1, x2, y2, angle, length])
        # print(line, length)
        if show_all_lines:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绘制所有直线（黑色）
    # 绘制最长的直线
    lengths.sort(key=lambda x: x[-1])
    longest_line = lengths[-1]
    x1, y1, x2, y2, angle, length = longest_line
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return angle
    # for i in range(len(lengths)-1,-1):
    #     line = lengths[i]
    #     x1, y1, x2, y2, angle, length = line
    #     if angle<-45 or angle>45:
    #         return 0
    #     else:
    #         return angle
    # longest_line = lengths[-1]
    # # print("longest_line: ", longest_line)
    # x1, y1, x2, y2, length = longest_line
    # if show_longest_line:
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制直线（红色）
    # # 计算这条直线的旋转角度
    # if x2 == x1:
    #     angle = 0
    # else:
    #     angle = math.atan((y2 - y1) / (x2 - x1))
    #     # print("angle-radin:", angle)  # 弧度形式
    #     angle = angle * (180 / math.pi)
    #     # print("angle-degree:", angle)  # 角度形式



def inpaint_with_ocr(ocr_data, img: np.ndarray):
    bboxes = [x[0] for x in ocr_data]
    ocr_img = img.copy()
    for box in bboxes:
        cv2.fillPoly(ocr_img, [np.array(box, np.int32)], (255, 255, 255))
    return ocr_img


def run(src_img):

    ocr_data = predict(src_img, ocr)
    ocr_img = inpaint_with_ocr(ocr_data, src_img)
    ret, dst = cv2.threshold(ocr_img, 150, 230, cv2.THRESH_TOZERO_INV)
    dst = 255 - dst
    angle = get_rotation_angle(dst, False, False)
    imag = rotate_bound(src_img, angle)
    return imag


def run2(src_img):
    img = src_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(img, kernel,iterations=5)
    erode = cv2.erode(img, kernel)
    res = cv2.absdiff(dilate, erode)
    retval, result = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)
    result = cv2.dilate(result, kernel)
    # cnts, _ = cv2.findContours(result.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # boxes = []
    # for cnt in cnts:
    #     rect = cv2.minAreaRect(cnt)
    #     box = cv2.boxPoints(rect)  # cv.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点
    #     box = np.int0(box)
    #     cv2.drawContours(img, [box], -1, (0, 0, 0), -1)
    #     boxes.append(box)
    # angle = get_rotation_angle(img, True, False)
    angle = get_rotation_angle(result)
    print(angle)
    # out_im = cv2.resize(np.concatenate((img, src_img), axis=1), (960, 480))
    # cv2.imshow(f'{angle}', out_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    imag = rotate_bound(src_img, angle)
    return imag


def get_rotate_imgs():
    src_img_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation'
    output_img_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation_rotated_opencv'
    os.makedirs(output_img_dir, exist_ok=True)
    imgs_list = glob.glob(f'{src_img_dir}/*.png')
    for i in tqdm(range(len(imgs_list))):
        src_img_path = imgs_list[i]
        _, filename = os.path.split(src_img_path)
        output_filepath = os.path.join(output_img_dir, filename)
        src_img = cv2.imread(src_img_path)
        imag = run2(src_img)
        cv2.imwrite(output_filepath, imag)


def merge_img(img1, img2):
    h, w, c = img1.shape
    img2 = cv2.resize(img2, (w, h))
    out_img = np.concatenate((img1, img2), axis=1)
    pt1 = [0, h // 2]
    pt2 = [2 * w, h // 2]
    cv2.line(out_img, pt1, pt2, (255, 0, 0), 3)
    out_img = cv2.resize(out_img, (960, 480))
    return out_img


def compare_imgs():
    img1_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation_rotated'
    img2_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation_rotated_opencv'
    imgs = os.listdir(img1_dir)
    imgs.sort()
    for i in tqdm(range(len(imgs))):
        filename = imgs[i]
        img1_path = os.path.join(img1_dir, filename)
        img2_path = os.path.join(img2_dir, filename)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        out_img = merge_img(img1, img2)
        cv2.imshow('win', out_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    # get_rotate_imgs()
    compare_imgs()
