import numpy as np
import cv2
import math
import os
import glob
import json


def get_table_imgs(img, label_data):
    tables = []
    shapes = label_data['shapes']
    for shape in shapes:
        shape_type = shape['shape_type']
        if shape_type == 'rectangle':
            points = shape['points']
            p1, p2 = np.int0(points)
            try:
                img_table = img[p1[1]:p2[1], p1[0]:p2[0]]
                if img_table.shape[0] > 0 and img_table.shape[
                        1] > 0 and img_table.shape[2] == 3:
                    tables.append(img_table)
            except:
                pass
        else:
            print('not rectangle!')
    return tables


def get_rotation_angle(image, show_longest_line=True, show_all_lines=False):
    # image = image.copy()  # 复制备份，因为line（）函数为in-place
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray, 100, 150, apertureSize=3)  # canny， 便于hough line减少运算量
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=60)  # 参数很关键
    # minLineLengh(线的最短长度，比这个短的都被忽略)
    # maxLineCap（两条直线之间的最大间隔，小于此值，认为是一条直线）
    # 函数cv2.HoughLinesP()是一种概率直线检测，原理上讲hough变换是一个耗时耗力的算法，
    # 尤其是每一个点计算，即使经过了canny转换了有的时候点的个数依然是庞大的，
    # 这个时候我们采取一种概率挑选机制，不是所有的点都计算，而是随机的选取一些个点来计算，相当于降采样。
    lengths = []  # 存储所有线的坐标、长度
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = ((x1 - x2)**2 + (y1 - y2)**2)**0.5  # 勾股定理，求直线长度
        lengths.append([x1, y1, x2, y2, length])
        # print(line, length)
        if show_all_lines:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 绘制所有直线（黑色）
    # 绘制最长的直线
    lengths.sort(key=lambda x: x[-1])
    longest_line = lengths[-1]
    # print("longest_line: ", longest_line)
    x1, y1, x2, y2, length = longest_line
    if show_longest_line:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 绘制直线（红色）
        cv2.imshow("longest", image)
    # 计算这条直线的旋转角度
    if x2 == x1:
        angle = 0
    else:
        angle = math.atan((y2 - y1) / (x2 - x1))
        # print("angle-radin:", angle)  # 弧度形式
        angle = angle * (180 / math.pi)
        # print("angle-degree:", angle)  # 角度形式
    return angle


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
        image, M, (nW, nH), borderValue=(255,255,255))  #BORDER_REPLICATE)
    # borderMode=cv2.BORDER_REPLICATE 使用边缘值填充
    # 或使用borderValue=(255,255,255)) # 使用常数填充边界（0,0,0）表示黑色


def run():
    img_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/images'
    label_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableDetection-labelme'
    output_dir = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation'
    os.makedirs(output_dir, exist_ok=True)
    xml_files = glob.glob(f'{label_dir}/*.json')
    # print(len(xml_files))
    import tqdm
    for i in tqdm.tqdm(range(len(xml_files))):
        xml_file = xml_files[i]
        _, filename = os.path.split(xml_file)
        filename, _ = os.path.splitext(filename)
        img_path = os.path.join(img_dir, f'{filename}.png')
        img = cv2.imread(img_path)
        with open(xml_file, 'r') as f:
            data = json.load(f)
            img_tables = get_table_imgs(img, data)
            for idx, img in enumerate(img_tables):
                angle = get_rotation_angle(img, False, False)
                imag = rotate_bound(img, angle)  # 关键
                output_file_path = os.path.join(output_dir,
                                                f'{filename}-{idx}.png')
                cv2.imwrite(output_file_path, imag)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)


if __name__ == '__main__':
    run()
