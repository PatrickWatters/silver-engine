"""Main script for training, validation, and inference."""

# Imports Python builtins.
import resource

# Imports PyTorch packages.
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning_fabric.utilities.seed import seed_everything as new_seed_everything

# Imports other packages.
from configargparse import Parser
from PIL import ImageFile

# Imports local packages.
from args import parse_args
from super import super_loader
from supermodel import SuperLightningModel

import os
# Prevents PIL from throwing invalid error on large image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Prevents DataLoader memory error.
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

def load_model(args):
    """Loads WS-DETR model and optionally loads weights."""
    model = SuperLightningModel(args=args,weights=None)
    return model

def load_trainer(args):
    """Loads PyTorch Lightning Trainer with callbacks."""

    # Instantiates progress bar. Changing refresh rate is useful when
    # stdout goes to a logfile (e.g., on cluster). 1 is normal and 0 disables.
    progress_bar = TQDMProgressBar(refresh_rate=args.refresh_rate)

    # Sets DDP strategy for multi-GPU training.
    args.strategy = "ddp" if args.gpus > 1 else None

    callbacks = [progress_bar]
    
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    return trainer

def main(args):
    """Trains, tests, with model as specified by args."""
    # Sets global seed for reproducibility.
    # Note: Due to CUDA operations which cannot be made deterministic,
    # the code will still not be perfectly reproducible.
    new_seed_everything(seed=42, workers=True)
    args.batch_size = None
    args.jobid = os.getpid()

    # Instantiates super dataloaders
    if args.task == "train":
        train_loader = super_loader(args, task="train")
        #val_loader = super_loader(args, task="val")
           
    elif args.task == "test":
        val_loader = super_loader(args, task="test")
    
    model = load_model(args)
    trainer:Trainer = load_trainer(args)

    if args.task == "train":
        trainer.fit(model, train_loader)
    elif args.task == "test":
        trainer.test(model, val_loader)
        
if __name__ == "__main__":
    args = parse_args(default_config_file='/Users/patrickwatters/Projects/super/torch/cfgs/quicktest.yaml')
    main(args)