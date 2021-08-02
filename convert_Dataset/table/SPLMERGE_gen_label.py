import numpy as np
import cv2
import json
import os
from scipy import stats

def get_mask(path):
        f = open(path, 'r')
        config = json.load(f)
        shapes = config['shapes']
        imageHeight = config['imageHeight']
        imageWidth = config['imageWidth']
        mask = np.zeros((imageHeight, imageWidth))
        for shape in shapes:
            points = np.array(shape['points'], np.int32)
            cv2.fillPoly(mask, [points], (255))
        h, w = mask.shape
        row = np.zeros(h)
        col = np.zeros(w)
        for i in range(h):
            row[i] = stats.mode(mask[i, :])[0][0]
        for j in range(w):
            col[j] = stats.mode(mask[:, j])[0][0]
        return mask, row, col
if __name__ == '__main__':
    lableme_dir = ''
    output_mask_dir = ''
    
    path = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation_lableme/0010-0.json'
    output_dir = './output_SPELERGE'
    os.makedirs(output_dir ,exist_ok=True)
    mask, row,col = get_mask(path)
    row = '\n'.join([str(int(x)) for x in row])
    # col = '\n'.join(col)
    with open(os.path.join(output_dir,'0010-0_row.txt'),'w') as f:
        f.write(row)
    