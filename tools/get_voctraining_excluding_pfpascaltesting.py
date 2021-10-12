from pycocotools.coco import COCO
import numpy as np
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument("--voc-ann", type=str, default="data/voc/voc_2012_train_aug_cocostyle.json")
    parser.add_argument("--pfpascal", type=str, default="data/PF-PASCAL/trn_pairs.csv")
    parser.add_argument("--output", type=str, default="data/voc/voc_2012_train_aug_excluding_pfpascal.json")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    coco = COCO(args.voc_ann)
    trn_pairs = np.loadtxt(args.pfpascal, delimiter=",", dtype=str)
    trn_pairs = trn_pairs[1:]
    trn_list = []
    for trn_pair in trn_pairs:
        trn_list.append(trn_pair[0][trn_pair[0].rfind('/')+1:])
        trn_list.append(trn_pair[1][trn_pair[1].rfind('/')+1:])
    trn_set = set(trn_list)
    img_list = []
    img_ids = []
    for image in coco.dataset['images']:
        if image['file_name'] not in trn_set:
            img_list.append(image)
            img_ids.append(image['id'])
    img_ids_set = set(img_ids)
    ann_list = []
    for ann in coco.dataset['annotations']:
        if ann['image_id'] in img_ids_set:
            ann_list.append(ann)

    coco.dataset['images'] = img_list
    coco.dataset['annotations'] = ann_list

    with open(args.output, "w") as f:
        json.dump(coco.dataset, f)