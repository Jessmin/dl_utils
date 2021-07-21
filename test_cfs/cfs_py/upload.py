from config import headers, cfs
import requests
import numpy as np
import json
import base64
from oss2 import SizedFileAdapter, determine_part_size
from urllib.parse import unquote, quote


def init_multipart_upload(filename, part_count):
    params = {
        'key': f'dlocr/{filename}',
        'partNumbers': [(x + 1) for x in range(part_count)]
    }
    result = requests.post(
        cfs.base_url + '/file/oss/initiateMultipartUpload',
        headers=headers,
        data=json.dumps(params)).json()
    return result


def upload_part(upload_url, img_bytes):
    headers = {}
    headers['Content-Type'] = 'application/octet-stream'
    # headers['Content-encoding'] = None
    requests.put(upload_url, headers=headers, data=img_bytes)
    print(f'upload successfully')


def comlete_mutipart_upload(headers, params):
    r = requests.post(
        cfs.base_url + '/file/oss/completeMultipartUpload',
        data=json.dumps(params),
        headers=headers)
    return r.json()


def list_parts(upload_id, key):
    url = cfs.base_url + '/file/oss/listParts'
    params = {'uploadId': str(upload_id), 'key': key}
    result = requests.get(url, headers=headers, params=params)
    return result.json()


def mutipart_upload(encoded_image, filename, total_size):
    part_size = 1024 * 1024 
    part_count = int(np.ceil(total_size / part_size))
    print(f'part_count:{part_count}')
    # step1.初始化分片
    result = init_multipart_upload(filename, part_count)
    print(result)
    if result['status'] == 1:
        raise Exception(f"{result['subMessage']}")
    else:
        data = result['data']
        partList = data['partList']
        encodedKey = data['encodedKey']
        uploadId = data['uploadId']
        myKey = base64.b64decode(encodedKey).decode('utf-8')
        offset = 0
        for part in partList:
            num_to_upload = min(part_size, total_size - offset)
            img_byte = SizedFileAdapter(encoded_image, num_to_upload)
            url = part['url']
            upload_part(url, img_byte)
            offset += num_to_upload
        # end upload part
        res = list_parts(uploadId, myKey)
        if res['status'] == 1:
            raise Exception(f'{res["subMessage"]}')
        else:
            if len(res['data']) != part_count:
                raise Exception('分片未完全上传成功')
        # end
        fileRecord = {
            "fileChannel": "OSS",
            "fileKey": myKey,
            "fileSize": total_size,
            "fileType": "png",
            "originName": filename,
            "status": 1
        }
        req = {'key': myKey, 'uploadId': uploadId, 'fileRecord': fileRecord}
        result = comlete_mutipart_upload(headers, req)
        if result['status'] == 1:
            raise Exception(f'{result["subMessage"]}')
        else:
            print(result)


if __name__ == '__main__':
    import cv2
    import io
    filepath = '/home/zhaohj/Documents/dataset/test_task/test_imgs/0185.png'
    img = cv2.imread(filepath)
    _, encoded_image = cv2.imencode('.jpg', img)
    iobuf = io.BytesIO(encoded_image)
    filesize = len(encoded_image)
    # total_size = os.path.getsize(filepath)
    # import cv2
    # import sys
    # img = cv2.imread(filepath)
    # print(total_size)
    # _, encoded_image = cv2.imencode('.png', img)
    filename = 'test.png'
    mutipart_upload(iobuf, filename, filesize)
