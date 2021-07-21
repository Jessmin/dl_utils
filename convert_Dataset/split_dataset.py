# coding=utf-8

import argparse
import os
import os.path as osp
import glob
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset_dir', help='input annotated directory')
parser.add_argument('test_ratio', help='test set ratio', default=0.3)
parser.add_argument('--random_state', help='random seed ', default=100)
parser.add_argument('--mode', choices=['voc', 'seg'], default='voc')
args = parser.parse_args()


def split_VOC():
    voc_dir = osp.join(args.dataset_dir, 'VOC')
    annotationDir = osp.join(voc_dir, 'Annotations')
    if not osp.exists(annotationDir):
        print('annotation directory not exists:', annotationDir)
        sys.exit(1)
    outputDir = osp.join(voc_dir, 'ImageSets', 'Main')
    if not osp.exists(outputDir):
        os.makedirs(outputDir)
    train_file = osp.join(outputDir, 'train.txt')
    test_file = osp.join(outputDir, 'test.txt')
    train_txt_file = osp.join(voc_dir, 'train.txt')
    test_txt_file = osp.join(voc_dir, 'test.txt')
    if osp.exists(train_file):
        os.remove(train_file)
    if osp.exists(test_file):
        os.remove(test_file)
    if osp.exists(train_txt_file):
        os.remove(train_txt_file)
    if osp.exists(test_txt_file):
        os.remove(test_txt_file)
    total_files = glob.glob(osp.join(annotationDir, '*.xml'))
    total_files = [Path(o).stem for o in total_files]
    train_set, test_set = train_test_split(
        total_files,
        test_size=float(args.test_ratio),
        random_state=int(args.random_state))
    f_train = open(train_file, 'w')
    f_train_txt = open(train_txt_file, 'w')
    for file in train_set:
        line = f'./Annotations/{file}.xml ./JPEGImages/{file}.jpg\n'
        f_train_txt.write(line)
        line = f'{file}\n'
        f_train.write(line)
    f_train.close()
    f_train_txt.close()
    f_test = open(test_file, 'w')
    f_test_txt = open(test_txt_file, 'w')
    for file in test_set:
        line = f'./Annotations/{file}.xml ./JPEGImages/{file}.jpg\n'
        f_test_txt.write(line)
        line = f'{file}\n'
        f_test.write(line)
    f_test.close()
    f_test_txt.close()
    print(
        "split Completed. Number of Train Samples: {}. Number of Test Samples: {}"
        .format(len(train_set), len(test_set)))


def split_seg():
    sep = '$&$'
    seg_dir = osp.join(args.dataset_dir, 'Segmentation')
    annotationDir = osp.join(seg_dir, 'Annotations')
    total_files = glob.glob(osp.join(annotationDir, '*.png'))
    total_files = [Path(o).stem for o in total_files]
    train_set, test_set = train_test_split(
        total_files,
        test_size=float(args.test_ratio),
        random_state=int(args.random_state))
    train_file = osp.join(seg_dir, 'train.txt')
    test_file = osp.join(seg_dir, 'test.txt')
    if osp.exists(train_file):
        os.remove(train_file)
    if osp.exists(test_file):
        os.remove(test_file)
    f_train = open(train_file, 'w')
    suffix_list = ['png', 'jpg', 'jpeg', 'webp']
    for file in train_set:
        dest_filename = f'Annotations/{file}.png'
        for suffix in suffix_list:
            source_filename = osp.join(args.dataset_dir,
                                       f'images/{file}.{suffix}')
            if osp.exists(source_filename):
                fname = f'../images/{file}.{suffix}'
                line = f'{fname}{sep}{dest_filename}\n'
                f_train.write(line)
                break
    f_train.close()
    f_test = open(test_file, 'w')
    for file in test_set:
        dest_filename = f'Annotations/{file}.png'
        for suffix in suffix_list:
            source_filename = osp.join(args.dataset_dir,
                                       f'images/{file}.{suffix}')
            if os.path.exists(source_filename):
                fname = f'../images/{file}.{suffix}'
                line = f'{fname}{sep}{dest_filename}\n'
                f_test.write(line)
                break
    f_test.close()


def main():
    if not osp.exists(args.dataset_dir):
        print('directory not exists:', args.dataset_dir)
        sys.exit(1)
    if args.mode == 'voc':
        split_VOC()
    elif args.mode == 'seg':
        split_seg()
    else:
        print('do not support')


if __name__ == '__main__':
    main()
