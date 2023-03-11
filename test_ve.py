from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
from dataloaders import R2DataLoader
from tqdm import tqdm
import json
from sklearn import metrics

import torch
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast,GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import parse_args
from models.classifier import Classifier
from timm.utils import accuracy, AverageMeter
from swin_utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, \
    reduce_tensor
import datetime
import torch.nn.functional as F
import time
from optimizer import build_optimizer
from lr_scheduler import build_scheduler

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args, num_classes, logger=None):
    # Prepare model
    model = Classifier(args, logger=logger, n_classes=num_classes)
    model.to(args.device)

    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def calculate_metricx(preds, targets):
    auclist = []
    for i in range(preds[0].shape[-1]):
        fpr, tpr, thresholds = metrics.roc_curve(targets[:, i], preds[:, i], pos_label=1)
        auclist.append(metrics.auc(fpr, tpr))
    pred_labels = preds > 0.5
    confusion_matrix = metrics.multilabel_confusion_matrix(y_true=targets, y_pred=pred_labels)
    return np.array([x for x in auclist if not np.isnan(x)]), confusion_matrix


@torch.no_grad()
def test(args):
    """ Train the model """
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.exp_name))

    # train_loader = R2DataLoader(args, split='train', shuffle=True)
    # val_loader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_loader = R2DataLoader(args, split='test', shuffle=False)

    args, model = setup(args, logger=logger, num_classes=test_loader.num_classes)

    model = DDP(model)

    state_dict = torch.load(args.pretrained)['model']
    #logger.info(state_dict.keys())
    model.load_state_dict(state_dict, strict=True)

    model.zero_grad()


    max_auc, max_auc_test = 0, 0
    start_time = time.time()
    best_epoch, best_epoch_test = 0, 0
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    auc_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    predlist = []
    true_list = []

    image_ids = []
    predictions = []
    for idx, (image_id, images) in tqdm(enumerate(test_loader)):
        images = images.cuda(non_blocking=True)
        # labels = labels.cuda(non_blocking=True)
        image_ids.extend(image_id)
        # compute output
        with autocast(dtype=torch.float32):
            logits = model(images)

        torch.cuda.synchronize()
        predictions.extend(logits.detach().cpu().numpy())

    data = {}
    for i in range(len(image_ids)):
        data[image_ids[i]] = predictions[i]
    torch.save(data, 'test_pred_logits.pth')


def main():
    args = parse_args()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60), rank=0, world_size=args.n_gpu)
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)
    torch.backends.cudnn.benchmark = True

    # Model & Tokenizer Setup

    # Training
    test(args)


if __name__ == "__main__":
    main()
