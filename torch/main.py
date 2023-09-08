"""Main script for training, validation."""

# Imports Python builtins.
import resource
from configargparse import Parser
from PIL import ImageFile

# Imports PyTorch packages.
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Imports other packages.
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.model_helpers import get_torchvision_model

# Imports local packages.
from super_tools.args import parse_args
from super_tools.super import super_loader
import os
# Prevents PIL from throwing invalid error on large image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Prevents DataLoader memory error.
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

class SuperLightningModel(L.LightningModule):
    
    def __init__(self, args, weights=None):
        super().__init__()
        self.args = args
        self.arch = args.arch
        self.weights = weights
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.workers = args.workers
        #self.data_path = args.data_path
        #self.index_file_path = args.index_file_path
        self.model = get_torchvision_model(self.arch, weights=self.weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss_train = F.cross_entropy(output, target)
        self.log("train_loss", loss_train)
        return loss_train

    def eval_step(self, batch, batch_idx, prefix: str):
        images, target = batch
        output = self.model(images)
        loss_val = F.cross_entropy(output, target)
        self.log(f"{prefix}_loss", loss_val)
        return loss_val

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def train_dataloader(self):
        import torchvision as tv
        
        transforms = tv.transforms.Compose([tv.transforms.RandomResizedCrop(224), tv.transforms.ToTensor()])
        
        return super_loader(args=self.args,transforms = transforms, task='train')

    def val_dataloader(self):
        import torchvision as tv
        
        transforms = tv.transforms.Compose([tv.transforms.RandomResizedCrop(224), tv.transforms.ToTensor()])
        
        return super_loader(args=self.args,transforms = transforms, task='val')


    def test_dataloader(self):
        return self.val_dataloader()


def main(args):
    """Trains, tests, or infers with model as specified by args."""
    # Sets global seed for reproducibility.
    # Note: Due to CUDA operations which cannot be made deterministic,
    # the code will still not be perfectly reproducible.
    seed_everything(seed=42, workers=True)
    args.batch_size =None
    args.jobid = os.getpid()
    # Sets output directory.
    model = SuperLightningModel(
        args=args,
        weights=None
    )
    trainer = L.Trainer()

    print("Train Model")

    if args.task == "train":
        trainer.fit(model)
    elif args.task == "test":
        trainer.test(model)
        
if __name__ == "__main__":
    args = parse_args(default_config_file='/Users/patrickwatters/Projects/super/torch/cfgs/quicktest.yaml')
    main(args)