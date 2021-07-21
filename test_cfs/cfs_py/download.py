from config import headers, cfs
import requests
import json
import cv2
import base64
import numpy as np


fileId = 'fc24de20-2cc9-4727-9ebe-25cdbf0e2d20'

def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

def download():
    try:
        url = cfs.base_url + '/file/decrypt/oss/downLoad' + '?fileId=' + fileId
        re = requests.post(url, headers=headers).json()
        if re['status'] == 0:
            img_base64 = re['data']
            return img_base64
        else:
            print(f'download failed:{re}')
            return None
    except Exception as e:
        print(f'download image failed:{e}')
    
if __name__ == '__main__':
    img_data = download()
    # print(img_data)
    if img_data is not None:
        img = base64_to_cv2(img_data)
        cv2.imshow('img',img)
        cv2.waitKey(0)
    
    
