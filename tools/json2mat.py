from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import sys
import json
import numpy as np
from scipy.io import savemat
from IPython import embed


val_set = '/mnt/data/voc/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    with open(input, 'r') as f:
        objs = json.load(f)

    with open(val_set, 'r') as f:
        img_ids = map(lambda x: x[:-1], f.readlines())

    sorted(objs, key=lambda x: x['image_id'])

    m  = len(objs)
    i = 0
    d = {}

    for idx, image_idx in enumerate(img_ids):
        bboxes = []
        scores = []
        segms = []
        labels = []

        #from IPython import embed; embed()
        #image_idx = int(str(filter(lambda x: x >= '0' and x <= '9', image_idx)))

        while i < m and idx+1 == objs[i]['image_id']:
            image_id = objs[i]['image_id']
            category_id = objs[i]['category_id']
            #bbox = m[j]['bbox']
            score = objs[i]['score']
            width, height = objs[i]['segmentation']['size']
            segm = maskUtils.decode([objs[i]['segmentation']])
            segms.append(segm)
            #bboxes.append(bbox)
            scores.append(score)
            labels.append(category_id)

            i += 1

        if len(segms) > 0:
            #file_name = cfg.coco.loadImgs(image_id)[0]['file_name'][:-4]
            key = 'img_' + str(idx+1)

            segms = np.concatenate(segms, axis=2)
            scores = np.concatenate(scores, axis=None)
            labels = np.concatenate(labels, axis=None)

            d.update({key + "_masks": segms})
            d.update({key + "_scores": scores})
            d.update({key + "_labels": labels})
            d.update({key + "_name": key})
        else:
            print(idx)

    savemat(output, d)
