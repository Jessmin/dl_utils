import numpy as np
import cv2
from PIL import Image, ImageFont,ImageDraw


def paint_chinese_opencv(im, chinese, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 25)
    fillColor = color  #(255,0,0)
    position = pos  #(100,100)
    # if not isinstance(chinese, unicode):
    # chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


img_l = cv2.imread(
    '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/images/0010.png')
img1 = cv2.imread('img.png')
img1 = paint_chinese_opencv(img1, '表格结构识别结果',(img1.shape[0] // 2, img1.shape[1] // 4), (255, 0, 0))

img2 = cv2.imread(
    '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation_rotated/0010-0.png'
)
img2 = paint_chinese_opencv(img2, '表格检测结果', (img2.shape[0] // 2, img2.shape[1] // 4), (255, 0, 0))
img_r = np.concatenate((img2, img1), axis=0)
img_l = cv2.resize(img_l, (img_r.shape[1], img_r.shape[0])) 
img_l = paint_chinese_opencv(img_l, '原始图片',(img_l.shape[0] // 4, img_l.shape[1] // 2), (255, 0, 0))
img_excel = cv2.imread('test.bmp')
img_excel = cv2.resize(img_excel, (img_r.shape[1], img_r.shape[0])) 
img_excel = paint_chinese_opencv(img_excel, '生成最终表格',(img_excel.shape[0] // 4, img_excel.shape[1] // 2), (255, 0, 0))
img = np.concatenate((img_l, img_r, img_excel), axis=1)

# cv2.imshow('img', img)
# cv2.waitKey(0)
cv2.imwrite('ocr_table_flow.png',img)