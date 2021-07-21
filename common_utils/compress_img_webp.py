# 编码
import webp
import glob
from PIL import Image
import os
"""
webp图片压缩测试
"""
img_dir = '/home/zhaohj/Documents/dataset/compare/test_img'
out_dir = '/home/zhaohj/Documents/dataset/compare/test_img_webp'
files = glob.glob(f'{img_dir}/*.png')
for file in files:
    _, filename = os.path.split(file)
    filename = filename.replace('.png','.webp')
    output_filepath = os.path.join(out_dir, filename)
    im = Image.open(file)  #读入文件
    webp.save_image(im, output_filepath, quality=100)  #压缩
