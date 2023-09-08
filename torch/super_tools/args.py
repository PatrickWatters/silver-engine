# Imports other packages.
from pytorch_lightning import Trainer
import configargparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))


def parse_args(default_config_file):
    """Parses command line and config file arguments."""

    parser = configargparse.ArgumentParser(
        description='Testing out serverless data loader on CPU or GPU',
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=False,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        default_config_files=[default_config_file])
    
    
    parser = add_input_args(parser)

    #parser = Trainer.add_argparse_args(parser)
    #parser = WS_DETR.add_model_specific_args(parser)
    args = parser.parse_args()
    
    return args

def add_input_args(parser):
    """Adds arguments not handled by Trainer or model."""
    parser.add(
        "--gprc_port",
        default='50052',
        help="port for gprc server.",
    )
    parser.add(
        "--arch",
        default='resnet18',
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)"
        )
    
    parser.add(
        "--weight_decay",
        type=float,
        help="Weight decay factor.",
    )

    parser.add(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum.",
    )

    parser.add(
        "--lr",
        type=float,
        default=0.1,
        help="initial learning rate",
    )

    parser.add(
        "--save_dir",
        default="out",
        help="Directory to save images and checkpoints; overrides PT dir.",
    )

    parser.add(
        "--task",
        choices=["train", "test", "infer"],
        help="Mode to run the model in.",
    )  
    parser.add(
        "--sampler",
        action="store_true",
        help="Whether to use a balanced random sampler in DataLoader.",
    )

    parser.add(
        "--workers",
        type=int,
        help="Number of workers in DataLoader.",
    )
    parser.add(
        "--gpus",
        type=int,
        help="Number of workers in DataLoader.",
    )
    parser.add(
        "--lr_step_size",
        type=int,
        help="How many epochs to run before dropping the learning rate.",
    )

    parser.add(
        "--pin_memory",
        action="store_true",
        help=(
            "pin_memory."
        ),
    )
    parser.add(
        "--max_epochs",
        type=int,
        help="Number of images per batch.",
    )
    parser.add(
        "--refresh_rate",
        type=int,
        help="Batch interval for updating training progress bar.",
    )
    
    return parser