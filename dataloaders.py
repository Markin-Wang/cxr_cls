import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset,IuxrayMultiImageClsDataset,MimiccxrSingleImageClsDataset
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import random

class R2DataLoader(DataLoader):
    def __init__(self, args, split, shuffle, vis=False):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.split = split
        self.drop_last = True if split =='train' else False
        self.vis = vis
        self.test = args.test
        if split != 'train' and self.batch_size>200:self.batch_size //= 2
        g = torch.Generator()
        g.manual_seed(args.seed)

        if split == 'train':
            # self.transform = transforms.Compose([
            #     transforms.Resize(256),
            #     transforms.RandomCrop(224),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomRotation(10),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.485, 0.456, 0.406),
            #                          (0.229, 0.224, 0.225))])
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageClsDataset(self.args, self.split, transform=self.transform, vis = self.vis)
        elif self.dataset_name.startswith('mimic'):
            self.dataset = MimiccxrSingleImageClsDataset(self.args, self.split, transform=self.transform, vis = self.vis)
            #self.dataset = MimiccxrSingleImageDataset(self.args, split=self.split, transform=self.transform)
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        self.sampler = torch.utils.data.DistributedSampler(
            self.dataset, num_replicas=num_tasks, rank=global_rank, shuffle=self.shuffle
        )

        self.num_classes = self.dataset.num_classes


        self.init_kwargs = {
            'dataset': self.dataset,
            'sampler': self.sampler,
            'batch_size': self.batch_size,
            #'shuffle':shuffle,
            'worker_init_fn': seed_worker,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': self.drop_last,
            'prefetch_factor': self.batch_size // self.num_workers * 2
        }


        # num_tasks = dist.get_world_size()
        # global_rank = dist.get_rank()
        #
        # self.sampler = DistributedSampler(self.dataset, num_replicas=num_tasks,
        #                                   rank=global_rank, shuffle=self.shuffle)

        super().__init__(**self.init_kwargs)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)