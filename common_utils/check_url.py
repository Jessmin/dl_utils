import pandas as pd
import requests
from tqdm import tqdm

df = pd.read_excel('/home/zhaohj/Documents/http/外链排查.xlsx')
urls = df['guide_file']
for i in tqdm(range(len(urls))):
    url = urls[i]
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            df['status'][i] = 'OK'
        else:
            df['status'][i] = ''
    except Exception as e:
        print(f'error in {i} e:{e}')
        continue
df.to_excel('外链排查.xlsx')