import cv2
import numpy as np
img =cv2.imread('/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation_rotated_opencv/0026-0.png')
img2 = cv2.Canny(img, 20, 250)
line = 4
minLineLength = 50
maxLineGap = 150
# HoughLinesP函数是概率直线检测，注意区分HoughLines函数
lines = cv2.HoughLinesP(img2, 1, np.pi / 180, 120, lines=line, minLineLength=minLineLength,maxLineGap=maxLineGap)
lines1 = lines[:, 0, :]  # 降维处理
# line 函数勾画直线
# (x1,y1),(x2,y2)坐标位置
# (0,255,0)设置BGR通道颜色
# 2 是设置颜色粗浅度
for x1, y1, x2, y2 in lines1:
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
cv2.imshow('img',img)
cv2.waitKey(0)

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# dilate = cv2.dilate(gray, kernel,iterations=5)
# erode = cv2.erode(gray, kernel)
# res = cv2.absdiff(dilate, erode)
# retval, result = cv2.threshold(res, 130, 255, cv2.THRESH_BINARY)
# result=cv2.dilate(result, kernel,iterations=2)
#
# cv2.imshow('result',result)
# cv2.waitKey(0)
# cnts, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# boxes = []
# for cnt in cnts:
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)  # cv.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点
#     box = np.int0(box)
#     if cnt.shape[0]<4:
#         continue
#     # cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
#     cv2.fillPoly(img, [np.array(box,np.int32)], (255, 255, 255))
#     # cv2.polylines(img, [np.array(box,np.int32)],True, (255，255，255),-1)
# cv2.imshow('img',img)
# cv2.waitKey(0)
