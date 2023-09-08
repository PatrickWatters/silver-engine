# Imports other packages.
from configargparse import Parser
import configargparse

def parse_args(default_config_file):
    """Parses command line and config file arguments."""

    parser = configargparse.ArgumentParser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=False,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        default_config_files=[default_config_file])
    
    parser = add_input_args(parser)
    args = parser.parse_args()
    return args

def add_input_args(parser):
    parser.add(
        "--train_data_dir",
        help="Training images directory.",
    )
    parser.add(
        "--val_data_dir",
        help="Training images directory.",
    )
    parser.add(
        "--data_dir",
        help="Training images directory.",
    )
    parser.add(
        "--dataset_droplast",
        action="store_true",
    )
    
    parser.add(
        "--log_dir",
        default="logs",
        help="Directory to save images and checkpoints; overrides PT dir.",
    )
    parser.add(
        "--sampler_seed",
        type=int,
        help="Number of images per batch.",
    )
    parser.add(
        "--batch_size",
        type=int,
        help="Number of images per batch.",
    )

    parser.add(
        "--source_system",
        choices=["s3", "local"],
        help="Source system",
    )
    parser.add(
        "--s3_bucket",
        help="Training images directory.",
    )

    parser.add(
        "--redis_host",
        help="Training images directory.",
    )
    
    parser.add(
        "--redis_port",
        help="Training images directory.",
    )
    parser.add(
        "--gprc_port",
        default='50052',
        help="port for gprc server.",
    )
    parser.add(
        "--dataload_lambda",
        help="Training images directory.",
    )
    parser.add(
        "--grpc_maxworkers",
        type=int,
        help="No. of gRPC server workers",
        default=10
    )
    parser.add(
        "--max_lookahead",
        type=int,
        default=25
    )
    parser.add(
        "--max_job_workers",
        type=int,
        default=25
    )
    
    return parser