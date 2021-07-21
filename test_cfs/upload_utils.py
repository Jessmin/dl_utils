import multiprocessing
import oss2
import requests

def get_block():
    pass


def upload(url, img_bytes, headers):
    headers={}
    headers['Content-Type'] = 'application/octet-stream'
    re = requests.put(url, headers=headers, data=img_bytes)
    print(re.content)


def uploadFileByPart(ossMultipartUploadResponse, part_count, part_size):
    pool = multiprocessing.Pool(processes=4)
    for i in range(4):
        url = ossMultipartUploadResponse['url'][i]
        startPos = i * part_size
        curPartSize = (i + 1 == part_count) if (
            localFile.length() - startPos) else part_size
        pool.map(upload, url)
    # end
    pool.close()
    pool.join()


if __name__ == '__main__':
    import numpy as np
    from io import BufferedReader, BytesIO
    import cv2
    img = cv2.imread('/home/zhaohj/Documents/dataset/signed_dataset/YMDS/images/0185.png')
    ret, img_encode = cv2.imencode('.png', img)
    f4 = BytesIO(img_encode)