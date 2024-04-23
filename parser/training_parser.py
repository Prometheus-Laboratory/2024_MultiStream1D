import argparse
from os.path import isdir, join, exists


def get_args():
    # parser definition
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_type",
                        type=str,
                        help="Which experiment to conduct ({cross_subject, within_subject})")
    parser.add_argument('dataset',
                        type=str,
                        help="Which dataset to use ({physionet, bci33a, bci41})")
    parser.add_argument("dataset_path",
                        type=str,
                        help="Path of the dataset")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Seed for reproducibility purposes (defaults to random)")
    parser.add_argument("--k_folds",
                        default=10,
                        type=int,
                        help="How many folds in the validation phase (defaults to 10)")
    parser.add_argument("--epochs",
                        default=50,
                        type=int,
                        help="How many epochs in the training phase (defaults to 50)")
    parser.add_argument("--batch_size",
                        default=2,
                        type=int,
                        help="Batch size for the training and validation phases (defaults to 2)")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help=f"Starting learning rate (defaults to {1e-4})")
    parser.add_argument("--early_stop",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help=f"Whether to early stop training after 5 epochs of not learning")
    parser.add_argument("--device",
                        default="auto",
                        type=str,
                        help="Device to use during training and validation ({auto, cpu, gpu, cuda}, defaults to auto)")
    parser.add_argument("--n_streams",
                        default=1,
                        type=int,
                        help=f"Number of convolutional streams (defaults to 1)")
    parser.add_argument("--n_streams_depth",
                        default=1,
                        type=int,
                        help=f"Number of blocks in each convolutional stream (defaults to 1)")
    parser.add_argument("--starting_conv_kernel_size",
                        default=1,
                        type=int,
                        help=f"Kernel size of the first stream, where the others have +2 increments (defaults to 1)")
    parser.add_argument("--n_classification_depth",
                        default=1,
                        type=int,
                        help=f"Number of linear layers at the end (defaults to 1)")
    parser.add_argument("--use_batchnorm",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help=f"Whether to use batch normalization into the conv blocks (defaults to False)")
    parser.add_argument("--learned_pooling",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help=f"Whether to use a convolution instead of a pooling layer at the end of each conv block "
                             f"(defaults to False)")
    parser.add_argument("--logs_dir",
                        default=join(".", "logs"),
                        type=str,
                        help="Where to store the logs (defaults to ./logs)")


    # parse args
    args = parser.parse_args()

    # asserts the validity of the data
    assert args.experiment_type in {"cross_subject", "within_subject", "cross", "within", "global", "specific"}
    if args.experiment_type in {"cross", "global"}:
        args.experiment_type = "cross_subject"
    if args.experiment_type in {"within", "specific"}:
        args.experiment_type = "within_subject"
    assert args.experiment_type in {"cross_subject", "within_subject"}

    assert args.dataset in {"physionet", "bci33a", "bci41"}
    assert isdir(args.dataset_path)

    assert args.k_folds >= 2
    assert args.epochs >= 1
    assert args.batch_size >= 1
    assert args.learning_rate > 0
    assert isinstance(args.early_stop, bool)

    assert args.n_streams >= 1
    assert args.n_streams_depth >= 1
    assert args.starting_conv_kernel_size >= 1
    assert args.n_classification_depth >= 1
    assert isinstance(args.use_batchnorm, bool)
    assert isinstance(args.learned_pooling, bool)

    assert args.device in {"auto", "cuda", "gpu", "cpu"}

    return args
