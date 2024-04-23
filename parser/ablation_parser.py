import argparse
from os.path import isdir, join, exists


def get_args():
    # parser definition
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',
                        type=str,
                        help="Which dataset to use ({physionet, bci33a, bci41})")
    parser.add_argument("dataset_path",
                        type=str,
                        help="Path of the dataset")
    parser.add_argument("--experiment_type",
                        default="all",
                        type=str,
                        help="Which experiment to conduct ({all, streams, classification, kernels, best}, defaults to 'all')")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Seed for reproducibility purposes (defaults to random)")
    parser.add_argument("--k_folds",
                        default=10,
                        type=int,
                        help="How many folds (defaults to 10)")
    parser.add_argument("--epochs",
                        default=50,
                        type=int,
                        help="How many epochs in the training phase (defaults to 50)")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help=f"Starting learning rate (defaults to {1e-4})")
    parser.add_argument("--early_stop",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help=f"Whether to early stop training after 10 epochs of not learning")
    parser.add_argument("--device",
                        default="auto",
                        type=str,
                        help="Device to use during training and validation ({auto, cpu, gpu, cuda}, defaults to auto)")
    parser.add_argument("--logs_dir",
                        default=join(".", "logs"),
                        type=str,
                        help="Where to store the logs (defaults to ./logs)")


    # parse args
    args = parser.parse_args()

    assert args.experiment_type in {"all", "streams", "classification", "kernels", "best"}
    assert args.dataset in {"physionet", "bci33a", "bci41"}
    assert isdir(args.dataset_path)

    assert args.k_folds >= 2
    assert args.epochs >= 1
    assert args.learning_rate > 0
    assert isinstance(args.early_stop, bool)

    assert args.device in {"auto", "cuda", "gpu", "cpu"}

    return args
