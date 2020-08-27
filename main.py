import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from model import PSPNet
from config import load_config
from preprocess import load_data
from dataset.prepare_voc import decode_semantic_label
from utils import SegmentationLosses, calculate_weigths_labels, SemanticSegmentationMetrics

import os
import numpy as np
import cv2


def save_checkpoint(model, optimizer, scheduler, args, global_step, scope=None):
    if scope is None:
        scope = global_step

    if args.distributed is False or args.local_rank == 0:
        if args.device_num > 1:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        torch.save({
            'model_state_dict': model_state_dict,
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join('checkpoint', 'checkpoint_model_{}.pth'.format(scope)))


class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.warmup_iteration = warmup_iteration

        super(PolyLr, self).__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        return base_lr * ((1 - (step / self.max_iteration)) ** (self.gamma))

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch)) for base_lr in
                    self.base_lrs]
        else:
            lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

        return lrs


def train(global_step, train_loader, model, optimizer, criterion, scheduler, args):
    model.train()
    metric = SemanticSegmentationMetrics(args)
    for data in train_loader:
        if global_step > args.max_iteration:
            break
        print('[Global Step: {0}]'.format(global_step), end=' ')
        img, label = data['image'], data['label']['semantic_logit']

        if args.cuda:
            for key in img.keys():
                img[key] = img[key].cuda()
            label = label.cuda()

        logit, loss = model(img, label.long())
        # loss = criterion(logit, label.long())
        evaluation = metric(logit, label, mode='train')
        print('[Loss: {0:.4f}], [Accuracy: {1:.5f}]'.format(loss.item(), evaluation['accuracy']))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if global_step % args.intervals == 0:
            save_checkpoint(model, optimizer, scheduler, args, global_step)
        global_step += 1
    return global_step


def eval(val_loader, model, args):
    metric = SemanticSegmentationMetrics(args)
    metric.clear()
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            img, label = data['image'], data['label']['semantic_logit']
            if args.cuda:
                for key in img.keys():
                    img[key] = img[key].cuda()
                label = label.cuda()

            logit = model(img)
            evaluation = metric(logit, label, 'val')
            print('[index: {0}], [Mean IoU: {2:.5f}], [Accuracy: {1:.5f}]'.format(idx + 1, evaluation['accuracy'], evaluation['mean_iou']))

            if args.result_save:
                if not os.path.isdir('result'):
                    os.mkdir('result')
                pred = logit.squeeze().detach().cpu().numpy()
                pred = decode_semantic_label(pred)
                pred *= np.stack([(label.squeeze().detach().cpu().numpy() != 255).astype(float)] * 3, axis=-1)
                cv2.imwrite('result/{}'.format(data['filename'][0]), pred.astype(np.uint8))


def main(args):
    train_loader, val_loader = load_data(args)
    # args.weight_labels = torch.tensor(calculate_weigths_labels('cityscape', train_loader, args.n_classes)).float()
    if args.cuda:
        # args.weight_labels = args.weight_labels.cuda()
        pass

    model = PSPNet()
    if args.cuda:
        model = model.cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)

    if args.evaluation:
        checkpoint = torch.load('./checkpoint/checkpoint_model_50000.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        eval(val_loader, model, args)
    else:
        # criterion = SegmentationLosses(weight=args.weight_labels, cuda=True)
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_mask, weight=None).cuda()

        backbone_params = nn.ParameterList()
        decoder_params = nn.ParameterList()

        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)

        params_list = [{'params': backbone_params},
                       {'params': decoder_params, 'lr': args.lr * 10}]

        optimizer = optim.SGD(params_list,
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay,
                              nesterov=True)
        scheduler = PolyLr(optimizer, gamma=args.gamma,
                           max_iteration=args.max_iteration,
                           warmup_iteration=args.warmup_iteration)

        global_step = 0
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        while global_step < args.max_iteration:
            global_step = train(global_step, train_loader, model, optimizer, criterion, scheduler, args)
        eval(val_loader, model, args)


if __name__ == '__main__':
    args = load_config()
    main(args)
