import argparse


def load_config():
    parser = argparse.ArgumentParser('Segmentation')

    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--n_classes', type=int, default=21)
    parser.add_argument('--ignore_mask', type=int, default=255)
    parser.add_argument('--dataset', type=str, default='voc2012')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--max_iteration', type=int, default=50000)
    parser.add_argument('--warmup_iteration', type=float, default=1500)
    parser.add_argument('--intervals', type=int, default=2000)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--result_save', type=bool, default=True)
    parser.add_argument('--evaluation', type=bool, default=False)
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    return args
