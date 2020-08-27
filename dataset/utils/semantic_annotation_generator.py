import os
import numpy as np
import tqdm
import json
import datetime

import sys
import argparse

from pathlib import Path
from PIL import Image

sys.path.append('../..')


class SemanticAnnotationGenerator:
    def __init__(self, image_dir, label_dir=None, target_dir=None, dataset_config_dir=None, tag='Custom',
                 split='train'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.tag = tag
        self.split = split

        if target_dir is None:
            self.target_dir = os.path.join(image_dir, '../annotations')
        else:
            self.target_dir = target_dir

        if os.path.isdir(self.target_dir) is False:
            os.makedirs(self.target_dir)

        self.annotation = {'info': dict(), 'licences': None, 'images': list(), 'annotations': list(),
                           'categories': list() if dataset_config_dir is not None else None}

        self.annotation['info']['description'] = self.tag + ' ' + 'Dataset'
        self.annotation['info']['version'] = '1.0'
        self.annotation['info']['year'] = datetime.datetime.now().year
        self.annotation['info']['date_created'] = str(datetime.datetime.now()) # .strftime("%Y-%m-%d %H:%M:%S.%s")

    def get_list(self, *args, **kwargs):
        self.image_list = os.listdir(os.path.join(self.image_dir, self.split))
        self.image_list.sort()

        if self.label_dir is not None:
            self.label_list = os.listdir(os.path.join(self.label_dir, self.split))
            self.label_list.sort()

    def run(self, *args, **kwargs):
        if self.label_dir is not None:
            iterator = tqdm.tqdm(enumerate(zip(self.image_list, self.label_list)),
                                 desc='Generating dataset list - ' + self.image_dir)
        else:
            iterator = tqdm.tqdm(enumerate(self.image_list),
                                 desc='Generating dataset list - ' + self.image_dir)

        for i, zipped in iterator:
            if self.label_dir is not None:
                image_file, label_file = zipped
            else:
                image_file = zipped
            # print(self.image_dir, self.split, image_file)
            image = Image.open(os.path.join(self.image_dir, image_file))

            image_annotation = dict()
            image_annotation['file_name'] = image_file
            image_annotation['width'] = image.size[0]
            image_annotation['height'] = image.size[1]
            image_annotation['id'] = i
            self.annotation['images'].append(image_annotation)

            if self.label_dir is not None:
                label_annotation = dict()
                label_annotation['file_name'] = label_file
                label_annotation['segments_info'] = None
                label_annotation['image_id'] = i
                self.annotation['annotations'].append(label_annotation)

        print(len(self.annotation['images']), "sample for", self.tag, self.split)

        json.dump(self.annotation, open(os.path.join(self.target_dir, self.split + '.json'), 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--label_dir', type=str)
    parser.add_argument('--target_dir', type=str, default=None)
    parser.add_argument('--dataset_config_dir', type=str)
    parser.add_argument('--tag', type=str)
    args = parser.parse_args()
    generator = SemanticAnnotationGenerator(image_dir=args.image_dir, label_dir=args.label_dir,
                                            target_dir=args.target_dir, dataset_config_dir=args.dataset_config_dir,
                                            tag=args.tag)
    generator.run()