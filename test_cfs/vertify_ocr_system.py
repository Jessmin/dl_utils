import uuid
import requests
import cv2
import io
import numpy as np
import json
import base64
from urllib.request import quote
import time


def mutipart_upload(encoded_image, filename, total_size, base_url, headers):
    part_size = 1024 * 1024
    part_count = int(np.ceil(total_size / part_size))
    print(f'part_count:{part_count}')
    # step1.初始化分片
    result = init_multipart_upload(filename, part_count, base_url, headers)
    if result['status'] == 0:
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
        res = list_parts(uploadId, myKey, base_url, headers)
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
        result = complete_mutipart_upload(headers, req, base_url)
        if result['status'] == 0:
            fileId = result['data']['downloadUrl']
            return fileId
        else:
            raise Exception(f'{result["subMessage"]}')
            return None
    else:
        raise Exception(f"{result['subMessage']}")
        return None


def init_multipart_upload(filename, part_count, base_url, headers):
    params = {
        'key': f'dlocr/{filename}',
        'partNumbers': [(x + 1) for x in range(part_count)]
    }
    result = requests.post(
        base_url + '/file/oss/initiateMultipartUpload',
        headers=headers,
        data=json.dumps(params),
        timeout=10)
    if result.status_code == 200:
        result = result.json()
    else:
        print(result.status_code)
    return result


def upload_part(upload_url, img_bytes):
    headers = {}
    headers['Content-Type'] = 'application/octet-stream'
    requests.put(upload_url, headers=headers, data=img_bytes, timeout=10)
    print('upload successfully')


def complete_mutipart_upload(headers, params, base_url):
    r = requests.post(
        base_url + '/file/oss/completeMultipartUpload',
        data=json.dumps(params),
        headers=headers)
    return r.json()


def list_parts(upload_id, key, base_url, headers):
    url = base_url + '/file/oss/listParts'
    params = {'uploadId': str(upload_id), 'key': key}
    result = requests.get(url, headers=headers, params=params, timeout=10)
    return result.json()


class SizedFileAdapter(object):
    """通过这个适配器（Adapter），可以把原先的 `file_object` 的长度限制到等于 `size`。"""

    def __init__(self, file_object, size):
        self.file_object = file_object
        self.size = size
        self.offset = 0

    def read(self, amt=None):
        if self.offset >= self.size:
            return ''

        if (amt is None or amt < 0) or (amt + self.offset >= self.size):
            data = self.file_object.read(self.size - self.offset)
            self.offset = self.size
            return data

        self.offset += amt
        return self.file_object.read(amt)

    @property
    def len(self):
        return self.size


class CfsUtils(object):
    def __init__(self, cfs_config):
        super(CfsUtils, self).__init__()
        self.customer_code = cfs_config['customer_code']
        self.product_code = cfs_config['product_code']
        self.secret_key = cfs_config['secret_key']
        self.file_channel = cfs_config['file_channel']
        self.base_url = cfs_config['base_url']
        self.headers = {
            'Content-Type': 'application/json',
            'Content-encoding': 'UTF-8',
            'customer-code': self.customer_code,
            'product-code': self.product_code,
            'secret-key': self.secret_key,
            'req-id': str(uuid.uuid1())
        }

    def download(self, fileId):
        try:
            url = self.base_url + '/file/decrypt/oss/downLoad' + '?fileId=' + fileId
            re = requests.post(url, headers=self.headers, timeout=10).json()
            if re['status'] == 0:
                img_base64 = re['data']
                return img_base64
            else:
                print(f'download failed:{re}')
                return None
        except Exception as e:
            print(f'download image failed:{e}')

    def upload(self, img_np: np.ndarray, filename: str):
        try:
            _, encoded_image = cv2.imencode('.jpg', img_np)
            iobuf = io.BytesIO(encoded_image)
            filesize = len(encoded_image)
            fileId = mutipart_upload(iobuf, filename, filesize, self.base_url,
                                     self.headers)
            return fileId
        except Exception as e:
            print(f'upload faild:{e}')
            return None


def send_request(req_url, img_url):
    data = {"images": [quote(img_url)]}
    headers = {'content-type': "application/json"}
    res = requests.post(url=req_url, headers=headers, data=json.dumps(data))
    return res


def run(output_file):
    req_url = 'http://127.0.0.1:8868/predict/rsi_system'
    image = np.ones((1920, 1080), dtype=np.uint8)
    cfs = {
        "customer_code": "yys",
        "product_code": "dlocr",
        "secret_key": "WRhd4D80BWRh",
        "file_channel": "oss",
        "base_url": "http://cfs.test.yaoyanshe.net:16201"
        # cfs.base_url = 'http://oss.uat.shiyantian.net'
    }
    # step1. upload image
    cfsUtil = CfsUtils(cfs)
    upload_url = cfsUtil.upload(image, 'test.png')
    # step2. send request
    response = send_request(req_url, upload_url)
    # step3. save result
    if response.status_code == 200:
        pass
    else:
        with open(output_file, 'a') as f:
            f.write(
                f'{time.asctime( time.localtime(time.time()) )} {response.content}'
            )


def timer(n):
    while True:
        run('output.log')
        time.sleep(n)


if __name__ == '__main__':
    # timer(10)
    run('output.log')
