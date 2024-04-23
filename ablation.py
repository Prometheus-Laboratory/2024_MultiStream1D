import json
import os
import random
import time
from datetime import datetime
from os.path import join, exists

import numpy as np

import torch

from torch.utils.data import Subset

from ablation.classification import ablation_classification_study
from ablation.best import ablation_best_study
from ablation.kernels import ablation_kernels_study
from ablation.streams import ablation_streams_study
from datasets.bci33a import BCICompetition3Dataset3a
from datasets.physionet import PhysioNetMotorImageryDataset
from parser import ablation_parser

# parses the arguments
args = ablation_parser.get_args()

starting_time = time.time()
print(
    f"Starting ablation study on {args.dataset} dataset at {datetime.fromtimestamp(int(starting_time)).time()}")

# sets up the logs dir
logs_dir = join(args.logs_dir, f"{args.dataset}_ablation_{str(int(starting_time))}")
if not exists(logs_dir):
    os.makedirs(logs_dir)

# sets the device to use
device = "cpu"
if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif args.device in {"cuda", "gpu"}:
    device = "cuda"

# sets the seed
seed = args.seed if args.seed else np.random.randint(low=0, high=1e6)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# saves metadata in the logs dir
metas = {
    "seed": seed,
    "device": device,
    "dataset": args.dataset
}
with open(join(logs_dir, "metas.json"), "w") as fp:
    json.dump(metas, fp, indent=True)

dataset_class, in_channels = None, None
if args.dataset == "physionet":
    dataset_class = PhysioNetMotorImageryDataset
elif args.dataset == "bci33a":
    dataset_class = BCICompetition3Dataset3a
elif args.dataset == "bci41":
    raise NotImplementedError
dataset = dataset_class(data_path=args.dataset_path)

# sets up k-fold
shuffled_indices = np.random.permutation(len(dataset))
fold_starting_indices = np.linspace(start=0, stop=len(dataset), num=args.k_folds + 1,
                                    endpoint=True, dtype=int)
folds = [shuffled_indices[i1:i2]
         for i1, i2 in zip(fold_starting_indices[:-1], fold_starting_indices[1:])]
train_indices, val_indices = np.concatenate(folds[:-1]), folds[-1]

if args.experiment_type in {"streams", "all"}:
    ablation_streams_study(dataset_train=Subset(dataset, train_indices),
                           dataset_val=Subset(dataset, val_indices),
                           in_channels=dataset.get_in_channels(),
                           epochs=args.epochs, device=device,
                           limit_train_batches=False,
                           logs_dir=logs_dir)

elif args.experiment_type in {"kernels", "all"}:
    ablation_kernels_study(dataset_train=Subset(dataset, train_indices),
                           dataset_val=Subset(dataset, val_indices),
                           in_channels=dataset.get_in_channels(),
                           epochs=args.epochs, device=device,
                           limit_train_batches=False,
                           logs_dir=logs_dir)

elif args.experiment_type in {"classification", "all"}:
    ablation_classification_study(dataset_train=Subset(dataset, train_indices),
                                  dataset_val=Subset(dataset, val_indices),
                                  in_channels=dataset.get_in_channels(),
                                  epochs=args.epochs, device=device,
                                  limit_train_batches=False,
                                  logs_dir=logs_dir)

elif args.experiment_type in {"best", "all"}:
    ablation_best_study(dataset_train=Subset(dataset, train_indices),
                        dataset_val=Subset(dataset, val_indices),
                        in_channels=dataset.get_in_channels(),
                        epochs=args.epochs, device=device,
                        limit_train_batches=False,
                        logs_dir=logs_dir)
