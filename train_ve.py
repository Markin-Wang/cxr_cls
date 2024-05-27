from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta
from dataloaders import R2DataLoader
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


def train_one_epoch(args, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, writer=None, scaler=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (imgs, label) in enumerate(data_loader):
        imgs = imgs.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        optimizer.zero_grad()
        with autocast(dtype=torch.float16):
            loss, logits = model(imgs, label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), label.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    writer.add_scalar('data/train_loss', loss_meter.avg)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def calculate_metricx(preds, targets):
    auclist = []
    for i in range(preds[0].shape[-1]):
        fpr, tpr, thresholds = metrics.roc_curve(targets[:, i], preds[:, i], pos_label=1)
        auclist.append(metrics.auc(fpr, tpr))
    pred_labels = preds > 0.5
    confusion_matrix = metrics.multilabel_confusion_matrix(y_true=targets, y_pred=pred_labels)
    return np.array([x for x in auclist if not np.isnan(x)]), confusion_matrix


@torch.no_grad()
def validate(args, data_loader, model):
    criterion = torch.nn.BCEWithLogitsLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    auc_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    predlist = []
    true_list = []

    end = time.time()
    for idx, (images, labels) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # compute output
        with autocast(dtype=torch.float16):
            loss, logits = model(images, labels)
            logits = torch.sigmoid(logits)

        torch.cuda.synchronize()
        if idx == 0:
            predlist = logits.cpu().numpy()
            true_list = labels.cpu().numpy()
        else:
            predlist = np.append(predlist, logits.cpu().numpy(), axis=0)
            true_list = np.append(true_list, labels.cpu().numpy(), axis=0)
        pred_labels = logits.ge(0.5)
        acc = (labels == pred_labels).float().sum() / (pred_labels.shape[-1] * pred_labels.shape[-2])
        loss = reduce_tensor(loss)
        acc1 = reduce_tensor(acc)
        loss_meter.update(loss.item(), labels.size(0))
        acc1_meter.update(acc1.item(), labels.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                # f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    auc, confusion_matrix = calculate_metricx(predlist, true_list)
    return acc1_meter.avg, auc, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def train(args):
    """ Train the model """
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.exp_name))

    # args.batch_size = args.batch_size // args.gradient_accumulation_steps


    train_loader = R2DataLoader(args, split='train', shuffle=True)
    # val_loader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_loader = R2DataLoader(args, split='test', shuffle=False)

    args, model = setup(args, logger=logger, num_classes=train_loader.num_classes)


    # if args.finetune:
    #     state_dict = torch.load(args.pretrained)['model']
    #     logger.info(state_dict.keys())
    #     state_dict.pop('head.weight')
    #     state_dict.pop('head.bias')
    #     model.load_state_dict(state_dict, strict=False)

    # Prepare optimizer and scheduler
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=args.learning_rate,
    #                             momentum=0.9,
    #                             weight_decay=args.weight_decay)
    t_total = 1
    # if args.decay_type == "cosine":
    #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # else:
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_scheduler(args, optimizer, len(train_loader))

    if args.local_rank != -1:
        model = DDP(model)
    scaler = GradScaler()

    # if args.fp16:
    #     model, optimizer = amp.initialize(models=model,
    #                                       optimizers=optimizer,
    #                                       opt_level=args.fp16_opt_level)
    #     amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    # if args.local_rank != -1:
    #     model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization epoch = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    # logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    losses = AverageMeter()
    global_step, max_accuracy = 0, 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    max_auc, max_auc_test = 0, 0
    start_time = time.time()
    best_epoch, best_epoch_test = 0, 0
    for epoch in range(args.epochs):
        # data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(args, model, criterion, train_loader, optimizer, epoch, None, lr_scheduler, writer, scaler)

        acc1, auc, loss = validate(args, test_loader, model)
        auc_score = auc.mean()
        writer.add_scalar('data/test_loss', loss)
        writer.add_scalar('data/test_acc', acc1)
        writer.add_scalar('data/test_auc_score', auc_score)
        writer.add_text('data/test_auc', str(auc))
        if auc_score > max_auc:
            max_auc = auc_score
            best_epoch = epoch
            save_checkpoint(args, epoch, model, max_auc, optimizer, lr_scheduler, logger)
        # if test_loader:
        #     acc1_test, auc_test, loss_test = validate(args, test_loader, model)
        #     auc_score_test = auc_test.mean()
        #     writer.add_scalar('data/test_loss', loss_test)
        #     writer.add_scalar('data/test_acc', acc1_test)
        #     writer.add_scalar('data/test_auc_score', auc_score_test)
        #     writer.add_text('data/test_auc', str(auc_test))
        #     if auc_score_test > max_auc_test:
        #         max_auc_test = auc_score_test
        #         best_epoch_test = epoch
        #         # save_checkpoint(config, args, epoch, model, max_auc, optimizer, scheduler, logger)

        logger.info('Auc for all classes: ' + ', '.join([str(round(x.item(), 5)) for x in auc]))
        logger.info(f' * Auc@1 {auc.mean():.3f}')
        logger.info(f' * Acc@1 {acc1:.3f} ')
        logger.info(f'Best model in epoch: {best_epoch}')

    logger.info(f"Auc of the network on the {len(test_loader)} test images: {max_auc:.5f}%")
    logger.info(f'Max auc: {max_auc:.5f}%')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    if args.local_rank in [-1, 0]:
        writer.close()
    # logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    logger.info(f"exp name:{args.exp_name}")


def main():
    args = parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = -1
            world_size = -1
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                             timeout=timedelta(minutes=60), rank=rank, world_size=args.n_gpu)
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
    train(args)


if __name__ == "__main__":
    main()