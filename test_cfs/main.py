import requests
import uuid
import json
from urllib.parse import unquote, quote
import numpy as np
from upload_utils import upload
import cv2
import base64

def upload_file(input_file: np.ndarray):
    part_size = 1024 * 1024
    file_length = input_file.size
    part_count = int(file_length / part_size)
    if file_length % part_size != 0:
        part_count += 1
    if part_count > 10000:
        raise Exception(f'Total parts count should not exceed 10000')
    params = {}
    key = 'test/auto/test.png'
    part_count = 2
    params = {'key': key, 'partNumbers': [x for x in range(part_count)]}
    result = requests.post(
        cfs.base_url + '/file/oss/initiateMultipartUpload',
        headers=headers,
        data=json.dumps(params)).json()
    if int(result['status']) == 1:
        raise Exception(f"{result['subMessage']}")
    else:
        if 'data' in result:
            result_list = result['data']['partList']
            if len(result_list) != part_count:
                raise Exception('分片未完全上传成功')
            data = result['data']
            print(data)
            if 'encodedKey' in result['data']:
                encodedKey = base64.b64decode(
                    data['encodedKey']).decode('utf-8')
                if 'uploadId' in data:
                    uploadId = data['uploadId']
                    uploadUrl = data['partList'][1]['url']
                    ret, encoded_image = cv2.imencode('.png', input_file)
                    img_bytes = encoded_image.tostring()
                    upload(uploadUrl, headers=headers, img_bytes=img_bytes)
                    #1.
                    req = {'key': encodedKey, 'uploadId': uploadId}
                    completeMultipartUpload(headers, req)
                    # if int(upload_result['status']) == 1:
                    #     raise Exception(upload_result['subMessage'])
                    # else:
                    #     data = upload_result['data']
                    #     print(data)


def completeMultipartUpload(headers, body):
    r = requests.post(
        cfs.base_url + '/file/oss/completeMultipartUpload',
        data=json.dumps(body),
        headers=headers)
    # return r.json()
    print(r.content)


def list_parts():
    url = cfs.base_url + '/file/oss/listParts'
    params = {}
    result = requests.post(url, headers=headers, params=json.dumps(params))
    print(result.content)


if __name__ == '__main__':
    img = np.zeros_like((33, 34, 3))
    upload_file(img)
