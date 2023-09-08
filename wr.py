import time

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
    

end = time.time()

total_data_load_time = AverageMeter("DataLoad", ":6.3f")
processing_time = AverageMeter('Process', ':6.3f')
time.sleep   (2)
total_data_load_time.update((time.time() - end))
processing_time.update((time.time() - end))

with open("file.txt", "a") as file:
    file.write(str(processing_time.val) + ',' + str(processing_time.sum) + ',' + str(processing_time.avg)
    + ','  + str(total_data_load_time.val) + ',' + str(total_data_load_time.sum) + ',' + str(total_data_load_time.avg) + "\n")