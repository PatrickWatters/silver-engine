import argparse
import json
import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from client import CMSClient
from sdl_dataset import SDLDataset
import random

class SDLSampler():

    def __init__(self,job_id, num_batches,sdl_client:CMSClient, ):        
        self.num_batches = num_batches
        self.sdl_client = sdl_client
        self.job_id = job_id
        self.epoch = 0

    def __iter__(self):
        for i in range(0, self.num_batches): 
            yield self.sdl_client.getBatches(self.job_id)
            #yield batch_id,btach_metadata,is_cached
    
    def __len__(self):
        return self.num_batches
    
    def set_epoch(self, epoch):
        self.epoch = epoch


model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("-a","--arch",metavar="ARCH",default="resnet18",choices=model_names,help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",)
parser.add_argument("-j", "--num-workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 4)")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("-epochs","--epochs", default=1, type=int, metavar="N", help="number of total epochs to run")  # default 90
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--wd","--weight-decay",default=1e-4,type=float,metavar="W",help="weight decay (default: 1e-4)",dest="weight_decay",)
parser.add_argument("-p", "--print-freq", default=1, type=int, metavar="N", help="print frequency (default: 1)")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--pin-memory", type=int, default=0)
best_acc1 = 0

def main():
    args = parser.parse_args()
    args.jobid = os.getpid()
    global best_acc1
    client = CMSClient()
    
    #copy model to the correct device
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("using CPU, this will be slow")
    
    if torch.cuda.is_available():
        if args.gpu:
            print("using cuda:'{}' device".format(args.arch))
            device = torch.device('cuda:{}'.format(args.gpu))
            model = model.cuda(args.gpu)
        else:
            print("using cuda device")
            device = torch.device("cuda")
            model = model.to(device)
    elif torch.backends.mps.is_available():
        print("using mps device")
        device = torch.device("mps")
        model = model.to(device)
    else:
        print("using CPU, this will be slow")
        device = torch.device("cpu")
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Data loading part
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [#transforms.Resize(256), 
         #transforms.CenterCrop(224), 
         transforms.ToTensor(), normalize]
    )

    #custom SDL Part

    response_message,successfully_registered,batches_per_epoch, dataset_len = client.registerJob(args.jobid)
    
    print(response_message)
    if not successfully_registered:
        exit()
        
    train_dataset = SDLDataset(job_id=args.jobid, length=dataset_len, transform=transform, target_transform=None)
    sdl_sampler = SDLSampler(job_id=args.jobid, num_batches=batches_per_epoch, sdl_client =client)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=None, num_workers=args.num_workers, sampler=sdl_sampler, pin_memory=True if args.pin_memory == 1 else False)
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train next epoch
        train(train_loader, model, criterion, optimizer, epoch, args,device, client)
    #client.job_ended_nofifcation(args.jobid)

def train(train_loader, model, criterion, optimizer, epoch, args,device, client:CMSClient):
    total_cache_hits = 0
    total_cache_misses = 0
    total_files = 0
    total_batch_time = AverageMeter("TotalTime", ":6.3f")
    data_fetch_time = AverageMeter("Fetch", ":6.3f")
    transfer_to_gpu_time = AverageMeter("Transfer", ":6.3f")
    processing_time = AverageMeter('Process', ':6.3f')
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    round_by_points = 3
    progress = ProgressMeter(
        len(train_loader), [total_batch_time, data_fetch_time,transfer_to_gpu_time,processing_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, labels,batch_id, cache_hit, prep_time) in enumerate(train_loader):
        #if i % 4 == 0:
        #    time.sleep(random.uniform(1.0, 5.0)) #delay data loading every 4 batches
        # measure data loading time
        data_fetch_time.update(time.time() - end)

        processing_started = time.time()
        time.sleep(0.012)
        processing_time.update(time.time() - processing_started)

        total_batch_time.update(time.time() - end)
        end = time.time()
    '''
    for i, (images, target,batch_id, cache_hit) in enumerate(train_loader):
        
        # measure data loading time
        data_fetch_time.update(time.time() - end)
        total_files += len(images)

        if cache_hit:
            total_cache_hits +=1
        else:
            total_cache_misses +=1

        #measure time to transfer data to GPU
        transfer_start = time.time()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        transfer_to_gpu_time.update(time.time() - transfer_start)

        #measure time to process batch on device
        processing_started = time.time()
        #time.sleep(args.training_speed)
     
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        processing_time.update(time.time() - processing_started)

        # measure elapsed time
        total_batch_time.update(time.time() - end)
    '''
class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # noqa
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n

        self.count += n
        self.avg = self.sum / self.count  # noqa

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30
    epochs."""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified
    values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()