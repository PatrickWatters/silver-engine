import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from client import CMSClient
from sdl_dataset import SDLDataset
from profiler_utils import DataStallProfiler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SDLSampler():

    def __init__(self,job_id, num_batches,sdl_client:CMSClient,):        
        self.num_batches = num_batches
        self.sdl_client = sdl_client
        self.job_id = job_id
        self.epoch = 0

    def __iter__(self):
        for i in range(0, self.num_batches): 
            yield self.sdl_client.getBatches(self.job_id)
            #yield i
    
    def __len__(self):
        return self.num_batches
    
    def set_epoch(self, epoch):
        self.epoch = epoch


model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("-a","--arch",metavar="ARCH",default="resnet18",choices=model_names,help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",)
parser.add_argument("-j", "--num-workers", default=2, type=int, metavar="N", help="number of data loading workers (default: 4)")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("-epochs","--epochs", default=3, type=int, metavar="N", help="number of total epochs to run")  # default 90
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--wd","--weight-decay",default=1e-4,type=float,metavar="W",help="weight decay (default: 1e-4)",dest="weight_decay",)
parser.add_argument("-p", "--print-freq", default=1, type=int, metavar="N", help="print frequency (default: 1)")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--pin-memory", type=int, default=0)
parser.add_argument("-tid","--trail-id", default=1, type=int, help="trialid")  # default 90 default=random.randint(0,100)

best_acc1 = 0
best_prec1 = 0

compute_time_list = []
data_time_list = []

def main():
    
    start_full = time.time()
    global best_prec1, args,best_acc1
    time_stat = []
    start = time.time()


    args = parser.parse_args()
    args.jobid = os.getpid()
    client = CMSClient()
    args.dprof = DataStallProfiler(args)

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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
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

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=None, 
                                               num_workers=args.num_workers, 
                                               sampler=sdl_sampler, 
                                               pin_memory=True if args.pin_memory == 1 else False)
    
    total_time = AverageMeter()
    dur_setup = time.time() - start
    time_stat.append(dur_setup)
    #print("Batch size for GPU {} is {}, workers={}".format(args.gpu, args.batch_size, args.workers)

    for epoch in range(args.start_epoch, args.epochs):
        # log timing
        start_ep = time.time()
        avg_train_time = train(train_loader, model, criterion, optimizer, epoch+1, args,device, client=client)
        total_time.update(avg_train_time)

        adjust_learning_rate(optimizer, epoch, args)
        # train next epoch
        
        #acc1 = validate(val_loader, model, criterion, args) # remember best acc@1 and save checkpoint

        scheduler.step()

        dur_ep = time.time() - start_ep
        print("EPOCH DURATION = {}".format(dur_ep))
        time_stat.append(dur_ep)
        #is_best = acc1 > best_acc1
        #best_acc1 = max(acc1, best_acc1)
    
    #client.job_ended_nofifcation(args.jobid)
    dur_full = time.time() - start_full
    print("Total time for all epochs = {}".format(dur_full))   

    args.dprof.stop_profiler()
    #profiler.gen_final_exel_report()

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
    
def train(train_loader, model, criterion, optimizer, epoch, args,device,client:CMSClient):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()
    end = time.time()
    args.dprof.start_data_tick()
    dataset_time = compute_time = 0


    for i, (images, labels,batch_id, cache_hit, prep_time) in enumerate(train_loader):

        # measure data loading time
        # measure data loading time
        data_time.update(time.time() - end)
        dataset_time += (time.time() - end)
        compute_start = time.time()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        #-----------------Stop data, start compute------#
        #if profiling, sync here
        #torch.cuda.synchronize()
        args.dprof.stop_data_tick()
        args.dprof.start_compute_tick()
        #-----------------------------------------------# 
       
        # compute output
        processing_started = time.time() 
        output = model(images)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        losses.update(to_python_float(loss.data), images.size(0))

        top1.update(to_python_float(acc1), images.size(0))
        top5.update(to_python_float(acc5), images.size(0))
       
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #torch.cuda.synchronize()

        #-----------------Stop compute, start data------#
        args.dprof.stop_compute_tick()
        args.dprof.start_data_tick()
        #-----------------------------------------------#
        compute_time += (time.time() - compute_start)

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

    data_time_list.append(dataset_time)
    compute_time_list.append(compute_time)
    return batch_time.avg

    

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