from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import os
import pylab

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
data_dir = '/home/zhaohj/Documents/dataset/Table/TableBank/TableBank/Detection'
annFile = f'{data_dir}/annotations/tablebank_latex_test.json'
coco = COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
catIds = coco.getCatIds(catNms=['table']);
imgIds = coco.getImgIds(imgIds=[1])
print(imgIds)
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

src_img = os.path.join(data_dir, f"images/{img['file_name']}")
import matplotlib.pyplot as plt

I = io.imread(src_img)
plt.imshow(I)
plt.axis('off')

plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()
