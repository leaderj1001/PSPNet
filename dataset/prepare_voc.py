import os
import sys

sys.path.append('../..')

from dataset.utils.semantic_annotation_generator import *


def encode_semantic_label(label, ignore_mask=255):
    color_map = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
        [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ]
    encoded_label = np.ones((*label.shape[:2],)) * ignore_mask
    for i in range(len(color_map)):
        encoded_label[np.all(label == color_map[i], axis=-1)] = i

    return encoded_label.astype(np.int32)


def decode_semantic_label(label):
    color_map = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
        [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
        [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ]

    if type(label) == np.ndarray:
        if len(label.shape) == 3:
            label = np.argmax(label, axis=0)

        decoded_label = np.zeros((*label.shape, 3))
        for i in range(len(color_map)):
            decoded_label[label == i] = color_map[i]

        return decoded_label

    elif type(label) == torch.Tensor:
        if len(label.shape) == 3:
            label = torch.argmax(label, dim=0)
        color_map = torch.tensor(color_map)

        decoded_label = torch.zeros((*label.shape, 3)).long()
        for i in range(len(color_map)):
            decoded_label[label == i] = color_map[i]

        return decoded_label


class Voc2012AnnotationGenerator(SemanticAnnotationGenerator):
    def get_list(self, root_dir):
        data_list = open(os.path.join(root_dir, 'ImageSets', 'Segmentation', self.split + '.txt'), 'r').read()
        data_list = data_list.split('\n')

        self.image_list = [i + '.jpg' for i in data_list if len(i) > 0]
        self.label_list = [i + '.png' for i in data_list if len(i) > 0]


class PrepareVoc2012:
    def __init__(self, root_dir):
        super(PrepareVoc2012, self).__init__()

        image_dir = os.path.join(root_dir, 'Images')
        if os.path.isdir(image_dir) is False:
            os.rename(os.path.join(root_dir, 'JPEGImages'), image_dir)

        label_dir = os.path.join(root_dir, 'Labels')
        if os.path.isdir(label_dir) is False:
            os.rename(os.path.join(root_dir, 'SegmentationClassAug'), label_dir)

        target_dir = os.path.join(root_dir, 'annotations')
        config_dir = os.path.join('dataset', 'dataset_configs', 'voc2012_config.json')

        self.train_anno_generator = Voc2012AnnotationGenerator(image_dir, label_dir, target_dir, config_dir, 'VOC2012', 'trainaug')
        self.train_anno_generator.get_list(root_dir)
        self.val_anno_generator = Voc2012AnnotationGenerator(image_dir, label_dir, target_dir, config_dir, 'VOC2012', 'val')
        self.val_anno_generator.get_list(root_dir)

    def run(self):
        self.train_anno_generator.run()
        self.val_anno_generator.run()


if __name__ == "__main__":
    # os.chdir('../../run')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./dataset/dataset_dir/VOC2012')
    args = parser.parse_args()

    preparation = PrepareVoc2012(args.root)
    preparation.run()