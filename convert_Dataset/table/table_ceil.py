import numpy as np
import cv2
import os
import json
from scipy import stats
from utils import minAreaRectbox, measure, eval_angle, draw_lines
from table_build import tableBuid, to_excel
from ocr_predict import predict
from iou import calculate_IOU

class table:
    def __init__(self):
        super(table, self).__init__()
        self.img = self.get_mask()
        h, w = self.img.shape[:2]
        self.adBoxes = [[0, 0, w, h]]
        self.table_ceil()
        self.table_build()

    def get_mask(self):
        path = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation_lableme/0010-0.json'
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
        row = np.zeros((h, w))
        col = np.zeros((h, w))
        for i in range(h):
            row[i, :] = stats.mode(mask[i, :])[0][0]
        # for j in range(w):
        #     col[:,j] = stats.mode(mask[:, j])[0][0]
        start_idx=0
        for j in range(w):
            val = stats.mode(mask[:, j])[0][0]
            if int(val)==255 and start_idx==0:
                start_idx=j
            elif int(val)==0 and start_idx!=0:
                end_idx = j
                mid_idx=(end_idx+start_idx)//2
                col[:,mid_idx-2:mid_idx+2]=255
                start_idx=0
            else:
                pass
        mask = np.zeros_like(row, dtype=np.uint8)
        mask[row > 0] = 255
        mask[col > 0] = 255
        return mask

    def table_ceil(self):
        tmp = self.img
        n = len(self.adBoxes)
        self.tableCeilBoxes = []
        self.childImgs = []
        for i in range(n):
            xmin, ymin, xmax, ymax = [int(x) for x in self.adBoxes[i]]
            childImg = self.img[ymin:ymax, xmin:xmax]
            labels = measure.label(tmp < 255, connectivity=2)  # 8连通区域标记
            regions = measure.regionprops(labels)
            ceilboxes = minAreaRectbox(regions, False, tmp.shape[1],
                                       tmp.shape[0], True, True)
            ceilboxes = np.array(ceilboxes)
            ceilboxes[:, [0, 2, 4, 6]] += xmin
            ceilboxes[:, [1, 3, 5, 7]] += ymin
            self.tableCeilBoxes.extend(ceilboxes)
            self.childImgs.append(childImg)

    def table_build(self):
        tablebuild = tableBuid(self.tableCeilBoxes)
        img = '/home/zhaohj/Documents/dataset/signed_dataset/YMDS/TableSegmentation_rotated/0010-0.png'
        src_img = cv2.imread(img)
        boxes, txts = predict(src_img)
        cor = tablebuild.cor
        for idx in range(len(cor)):
            out_val=''
            line = cor[idx]
            gt_box = self.tableCeilBoxes[idx]
            gt_box[gt_box<0]=0
            val=[]
            for i in range(len(boxes)):
                box = boxes[i]
                pt1 = [gt_box[0],gt_box[1],gt_box[4],gt_box[5]]
                pt2 =[box[0][0],box[0][1],box[2][0],box[2][1]]
                val.append([i,calculate_IOU(pt1,pt2)])
            val = sorted(val, key=lambda x: x[1])
            if val[-1][1]>0.8:
                out_val = txts[val[-1][0]]
            line['text'] = f'{out_val}'
        workbook = to_excel(cor, workbook=None)
        self.res = cor
        self.workbook = workbook

if __name__ == '__main__':
    tab=table()
    workbook = tab.workbook
    workbook.save('test'+'.xlsx')
    img= tab.img
    cv2.imwrite('img.png',img)