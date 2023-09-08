"""Defines dataset, dataloader, and processing for COCO-style datasets."""

# Imports Python builtins.
import json
from typing import (Callable,Optional,)
import base64
from PIL import Image
import io 
# Imports PyTorch packages.
import torch
from torch.utils.data import DataLoader
# Imports local packages.
from client import CMSClient
import os

class SDLSampler():

    def __init__(self,job_id,task, num_batches,sdl_client:CMSClient,):        
        self.num_batches = num_batches
        self.sdl_client = sdl_client
        self.job_id = job_id
        self.epoch = 0
        self.task = task

    def __iter__(self):
        for i in range(0, self.num_batches): 
            yield self.sdl_client.getBatches(self.job_id,self.mode)

    def __len__(self):
        return self.num_batches
    
    def set_epoch(self, epoch):
        self.epoch = epoch


class SuperImagenetDataset():
    def __init__(self,job_id:int,length:int, transform:Optional[Callable] =None,task="train"):
        aug = True if task == "train" else False
        self.transform = transform
        self.job_id = job_id
        self.lenth = length

    def __len__(self) -> int:
        return self.lenth
    
    def __getitem__(self, input):
        batch_data = input.data
        batch_id = input.batchid
        cache_hit = True
        torch_imgs, torch_targets = self.convert_json_batch_to_torch_format(batch_data)
        return torch_imgs, torch_targets

    def convert_json_batch_to_torch_format(self,batch_data):
        samples = json.loads(batch_data)
        images = []
        targets  =[]
        for image,target in samples:
            # Applies transformations to image and target.
            image = Image.open(io.BytesIO(base64.b64decode(image)))
            if self.transform is not None:
                image = self.transform(image)
            #if self.target_transform is not None:
            #    target = self.target_transform(target)
            images.append(image)
            targets.append(target)
        return torch.stack(images), torch.tensor(targets)


def super_loader(args, transforms=None, task='train'):
    """Builds a dataloader for SUPER style datasets."""
    #register job
    super_client = CMSClient()
    message,success,batches_per_epoch, dataset_length = super_client.registerJob(args.jobid)

    # Extracts data and labels location from args based on task.
    dataset = SuperImagenetDataset(job_id=args.jobid, length=dataset_length, transform=transforms, task=task)

    # Initializes a balanced random sampler for single-GPU training only.
    sampler = None
    sampler = SDLSampler(args.jobid, task=task, num_batches=batches_per_epoch,sdl_client=super_client)

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
    )
    return loader