import uuid


class CfsConfig(object):
    def __init__(self):
        super(CfsConfig, self)


cfs = CfsConfig()
cfs.customer_code = 'yys'
cfs.product_code = 'dlocr'
cfs.secret_key = 'WRhd4D80BWRh'
cfs.file_channel = 'oss'
cfs.base_url = 'http://cfs.test.yaoyanshe.net:16201'

headers = {
    'Content-Type': 'application/json',
    'Content-encoding': 'UTF-8',
    'customer-code': cfs.customer_code,
    'product-code': cfs.product_code,
    'secret-key': cfs.secret_key,
    'req-id': str(uuid.uuid1())
}