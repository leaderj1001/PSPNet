import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = np.array(sample['image']['original_scale']).astype(np.float32)
        image /= 255
        image -= self.mean
        image /= self.std

        sample['image']['original_scale'] = image

        return sample


class RandomGaussianBlur(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = Image.fromarray(sample['image']['original_scale'])
        if np.random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
        sample['image']['original_scale'] = np.array(image)

        return sample


class RandomEnhance(object):
    def __init__(self):
        self.enhance_method = [ImageEnhance.Contrast, ImageEnhance.Brightness, ImageEnhance.Sharpness]

    def __call__(self, sample):
        np.random.shuffle(self.enhance_method)
        image = Image.fromarray(sample['image']['original_scale'])

        for method in self.enhance_method:
            if np.random.random() > 0.5:
                enhancer = method(image)
                factor = float(1 + np.random.random() / 10)
                image = enhancer.enhance(factor)

        sample['image']['original_scale'] = np.array(image)
        return sample


class RandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']

        if np.random.random() < 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)

        sample['image']['original_scale'] = image
        sample['label']['semantic_logit'] = label

        return sample


class RandomScaleRandomCrop(object):
    def __init__(self, base_size, crop_size, scale_range=(0.5, 2.0), ignore_mask=255):

        if '__iter__' not in dir(base_size):
            self.base_size = (base_size, base_size)
        else:
            self.base_size = base_size

        if '__iter__' not in dir(crop_size):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

        self.scale_range = scale_range
        self.ignore_mask = ignore_mask

    def __call__(self, sample):
        image = Image.fromarray(sample['image']['original_scale'])
        label = Image.fromarray(sample['label']['semantic_logit'].astype(np.int32), mode='I')

        width, height = image.size
        scale = np.random.rand() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]

        if width > height:
            resize_height = int(scale * self.base_size[1])
            resize_width = int(width * (resize_height / height))

        else:
            resize_width = int(scale * self.base_size[0])
            resize_height = int(height * (resize_width / width))

        image = image.resize((resize_width, resize_height), Image.BILINEAR)
        label = label.resize((resize_width, resize_height), Image.NEAREST)

        padding = [0, 0]
        if resize_width < self.crop_size[0]:
            padding[0] = self.crop_size[0] - resize_width

        if resize_height < self.crop_size[1]:
            padding[1] = self.crop_size[1] - resize_height

        if np.sum(padding) != 0:
            image = ImageOps.expand(image, (0, 0, *padding), fill=0)
            label = ImageOps.expand(label, (0, 0, *padding), fill=self.ignore_mask)

        width, height = image.size
        crop_coordinate = np.array([np.random.randint(0, width - self.crop_size[0] + 1),
                                    np.random.randint(0, height - self.crop_size[1] + 1)])

        image = image.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))
        label = label.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))

        sample['image']['original_scale'] = np.array(image)
        sample['label']['semantic_logit'] = np.array(label)

        return sample


class FixedScaleCenterCrop(object):
    def __init__(self, crop_size):
        if '__iter__' not in dir(crop_size):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size

    def __call__(self, sample):
        image = Image.fromarray(sample['image']['original_scale'])
        label = Image.fromarray(sample['label']['semantic_logit'].astype(np.int32), mode='I')

        width, height = image.size

        if width > height:
            resize_height = int(self.crop_size[1])
            resize_width = int(width * (resize_height / height))

        else:
            resize_width = int(self.crop_size[0])
            resize_height = int(height * (resize_width / width))

        image = image.resize((resize_width, resize_height), Image.BILINEAR)
        label = label.resize((resize_width, resize_height), Image.NEAREST)

        crop_coordinate = np.array([int(resize_width - self.crop_size[0]) // 2,
                                    int(resize_height - self.crop_size[1]) // 2])

        image = image.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))
        label = label.crop((*crop_coordinate, *(crop_coordinate + self.crop_size)))

        sample['image']['original_scale'] = np.array(image)
        sample['label']['semantic_logit'] = np.array(label)

        return sample


class Resize(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']['original_scale']
        label = sample['label']['semantic_logit']

        image = F.interpolate(image.expand(1, *image.shape), scale_factor=self.scale, mode='bilinear', align_corners=False)
        label = F.interpolate(label.float().expand(1, 1, *label.shape), scale_factor=self.scale, mode='nearest')

        sample['image']['original_scale'] = image.squeeze(0)
        sample['label']['semantic_logit'] = label.squeeze().long()

        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.array(sample['image']['original_scale']).astype(np.float32)
        label = np.array(sample['label']['semantic_logit'])
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label)

        sample['image']['original_scale'] = image
        sample['label']['semantic_logit'] = label

        return sample


class MultiScale(object):
    def __init__(self, scale_list):
        self.scale_list = scale_list

    def __call__(self, sample):
        image = sample['image']['original_scale']
        images = dict()

        for scale in self.scale_list:
            if scale == 1:
                images['original_scale'] = image
            else:
                images[str(scale)] = F.interpolate(image.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
        sample['image'] = images

        return sample


class Flip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        images = sample['image']

        old_keys = list(images.keys())
        for key in old_keys:
            images.update({key + '_flip': torch.flip(images[key], dims=[-1])})

        sample['image'] = images
        return sample


# Reference
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/calculate_weights.py
def calculate_weigths_labels(dataset, dataloader, num_classes):
    classes_weights_path = os.path.join('weight', '{}_classes_weight_ratios.npy'.format(dataset))
    if os.path.isfile(classes_weights_path):
        return np.load(classes_weights_path)
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']['semantic_logit']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    np.save(classes_weights_path, ret)

    return ret


# Reference
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/loss.py
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SemanticSegmentationMetrics:
    def __init__(self, FLAGS):
        self.class_num = FLAGS.n_classes
        self.ignore_mask = FLAGS.ignore_mask
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))
        self.class_iou = []
        self.mean_iou = 0.0
        self.accuracy = 0.0

    def __call__(self, prediction, label, mode='train'):
        # prediction = prediction['semantic_logit']
        # label = label['semantic_logit']

        self.compute_confusion_matrix_and_add_up(label, prediction)
        if mode == 'train':
            accuracy = self.compute_pixel_accuracy()
            metric_dict = {'accuracy': accuracy}
        else:
            class_iou = self.compute_class_iou()
            mean_iou = self.compute_mean_iou()
            accuracy = self.compute_pixel_accuracy()
            metric_dict = {'mean_iou': mean_iou, 'accuracy': accuracy} #, 'class_iou': dict()}
            # for i, iou in enumerate(class_iou):
            #     metric_dict['class_iou']['class_' + str(i)] = iou
        return metric_dict

    def clear(self):
        self.confusion_matrix = np.zeros((self.class_num, self.class_num))

    def compute_confusion_matrix(self, label, image):
        if len(label.shape) == 4:
            label = torch.argmax(label, dim=1)
        if len(image.shape) == 4:
            image = torch.argmax(image, dim=1)

        label = label.flatten().cpu().numpy().astype(np.int64)
        image = image.flatten().cpu().numpy().astype(np.int64)

        valid_indices = (label != self.ignore_mask) & (0 <= label) & (label < self.class_num)

        enhanced_label = self.class_num * label[valid_indices].astype(np.int32) + image[valid_indices]
        confusion_matrix = np.bincount(enhanced_label, minlength=self.class_num * self.class_num)
        confusion_matrix = np.reshape(confusion_matrix, (self.class_num, self.class_num))

        return confusion_matrix

    def compute_confusion_matrix_and_add_up(self, label, image):
        self.confusion_matrix += self.compute_confusion_matrix(label, image)

    def compute_pixel_accuracy(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    def compute_class_iou(self):
        class_iou = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=0) + np.sum(self.confusion_matrix, axis=1) - np.diag(
                self.confusion_matrix))
        return class_iou

    def compute_mean_iou(self):
        class_iou = self.compute_class_iou()
        return np.nanmean(class_iou)
