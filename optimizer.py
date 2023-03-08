# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import json
from functools import partial
from torch import optim as optim


def build_optimizer(args, model):
    # ve_params = list(map(id, model.visual_extractor.parameters()))
    # ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    if args.optimizer == 'Adam':
        optimizer =optim.Adam(
            [{'params': model.transformer.parameters(), 'lr': args.learning_rate},
             {'params': model.head.parameters(), 'lr': args.mul_head * args.learning_rate}],
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'AdamW':
        optimizer =optim.AdamW(
            [{'params': model.model.parameters(), 'lr': args.learning_rate},
             {'params': model.head.parameters(), 'lr': args.mul_head * args.learning_rate}],
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(
            [{'params': model.transformer.parameters(), 'lr': args.learning_rate},
             {'params': model.head.parameters(), 'lr': args.mul_head * args.learning_rate}],
            lr = args.learning_rate,
            weight_decay=args.weight_decay,
            momentum = 0.9,
            nesterov=True,
        )
    else:
        optimizer = None
    return optimizer