import json


class RequestItem(object):
    def __init__(self, url='', fileId='', base64Str=''):
        self.url = url
        self.fileId = fileId
        self.base64Str = base64Str
        
        
with open('test.json', 'r', encoding='utf-8') as f:
    info = json.load(f)
    data = info['data']
    
    for d in data:
        reqItem = RequestItem()
        reqItem.__dict__ = d
        print(reqItem.url)
        print(reqItem.fileId)
        print(reqItem.base64Str)