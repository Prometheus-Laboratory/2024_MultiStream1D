import json
import logging
import os
import random
import re
import time
from datetime import datetime
from os import makedirs
from os.path import join, isdir, abspath, splitext, split, exists
from pprint import pprint

import numpy as np
import pandas as pd
import mne
from braindecode import EEGClassifier
from braindecode.util import set_random_seeds

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from scipy import signal
from torch import optim
from sklearn.metrics import f1_score, accuracy_score
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
from tqdm import tqdm

import models
from datasets.bci33a import BCICompetition3Dataset3a
from datasets.physionet import PhysioNetMotorImageryDataset
from models.bcinet import BCINet
from models.multi_stream_net import MultiStream1dNet
from parser import training_parser


def k_fold(dataset: Dataset, in_channels: int,
           k: int = 10, batch_size: int = 8, epochs: int = 100, early_stop: bool = False,
           device: str = "cpu", learning_rate: float = 1e-4,
           logs_dir: str = join(".", "logs"),
           logging_interval: int = 1, verbose: bool = True, save_models: bool = False,
           n_streams: int = 1, n_streams_depth: int = 1,
           starting_conv_kernel_size: int = 1, n_classification_depth: int = 1,
           use_batchnorm: bool = False, learned_pooling: bool = False,
           checkpoint_path: str = None):
    # eventually creates logging directory
    if not isdir(logs_dir):
        makedirs(logs_dir)
    # sets up k-fold
    shuffled_indices = np.random.permutation(len(dataset))
    fold_starting_indices = np.linspace(start=0, stop=len(dataset), num=k + 1,
                                        endpoint=True, dtype=int)
    folds = [shuffled_indices[i1:i2]
             for i1, i2 in zip(fold_starting_indices[:-1], fold_starting_indices[1:])]

    # loops over k folds
    for i_excluded_fold in range(len(folds)):
        train_indices = [i
                         for i_fold, f in enumerate(folds)
                         for i in f
                         if i_fold != i_excluded_fold]
        val_indices = [i
                       for i_fold, f in enumerate(folds)
                       for i in f
                       if i_fold == i_excluded_fold]

        dataloader_train = DataLoader(Subset(dataset, train_indices), shuffle=True,
                                      batch_size=batch_size, num_workers=os.cpu_count())
        dataloader_val = DataLoader(Subset(dataset, val_indices), shuffle=False,
                                    batch_size=batch_size, num_workers=os.cpu_count())

        # model
        if checkpoint_path is not None:
            model = MultiStream1dNet.load_from_checkpoint(checkpoint_path,
                                                          device=device, in_channels=in_channels,
                                                          adaptive_pooling_output_size=32, learning_rate=learning_rate,
                                                          n_streams=n_streams, n_streams_depth=n_streams_depth,
                                                          starting_conv_kernel_size=starting_conv_kernel_size,
                                                          classification_depth=n_classification_depth,
                                                          use_batchnorm=use_batchnorm, learned_pooling=learned_pooling)
        else:
            model = MultiStream1dNet(device=device, in_channels=in_channels,
                                     adaptive_pooling_output_size=32, learning_rate=learning_rate,
                                     n_streams=n_streams, n_streams_depth=n_streams_depth,
                                     starting_conv_kernel_size=starting_conv_kernel_size,
                                     n_classification_depth=n_classification_depth,
                                     use_batchnorm=use_batchnorm, learned_pooling=learned_pooling)

        # eventually disable logging
        if not verbose:
            logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

        # training
        callbacks = []
        if save_models:
            callbacks += [
                ModelCheckpoint(
                    dirpath=join(logs_dir, "checkpoints", f"fold_{i_excluded_fold}"),
                    filename="bcinet-{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}",
                    save_top_k=1, monitor="val_acc", mode="max"
                )
            ]
        if early_stop:
            callbacks += [
                EarlyStopping(monitor="val_acc",
                              min_delta=0, patience=10,
                              verbose=True, mode="max", check_on_train_epoch_end=False)
            ]
        trainer = pl.Trainer(gpus=1 if device == "cuda" else 0, precision=32, max_epochs=epochs,
                             # stochastic_weight_avg=True,
                             check_val_every_n_epoch=1, logger=False, num_sanity_val_steps=0,
                             # auto_lr_find=True,
                             enable_checkpointing=save_models,
                             callbacks=callbacks,
                             log_every_n_steps=logging_interval if logging_interval > 0 else -1,
                             enable_progress_bar=verbose, enable_model_summary=verbose)
        # trainer.tune(model, dataloader_train, dataloader_val)
        trainer.fit(model, dataloader_train, dataloader_val)

        # eventually restore logging
        if not verbose:
            logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

        # saves the stats for the fold
        model.get_stats().to_csv(join(logs_dir, f"fold_{i_excluded_fold}.csv"), index=False)


# parses the arguments
args = training_parser.get_args()

starting_time = time.time()
print(
    f"Starting {args.experiment_type} training on {args.dataset} dataset at {datetime.fromtimestamp(int(starting_time)).time()}")

# sets up the logs dir
logs_dir = join(args.logs_dir, f"{args.dataset}_{args.experiment_type}_{str(int(starting_time))}")
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

dataset_class, in_channels = None, None
if args.dataset == "physionet":
    dataset_class = PhysioNetMotorImageryDataset
elif args.dataset == "bci33a":
    dataset_class = BCICompetition3Dataset3a
elif args.dataset == "bci41":
    raise NotImplementedError

# saves metadata in the logs dir
metas = {
    "seed": seed,
    "dataset": args.dataset,
    "time_start": starting_time,
    "model": {
        "n_streams": args.n_streams,
        "n_streams_depth": args.n_streams_depth,
        "n_classification_depth": args.n_classification_depth,
        "adaptive_pooling_output_size": 32,
        "starting_conv_kernel_size": args.starting_conv_kernel_size,
        "use_batchnorm": args.use_batchnorm,
        "learned_pooling": args.learned_pooling,
        "in_channels": dataset_class.get_in_channels(),
        "device": device,
    }
}
with open(join(logs_dir, "metas.json"), "w") as fp:
    json.dump(metas, fp, indent=True)

if args.experiment_type == "cross_subject":
    k_fold(dataset=dataset_class(data_path=args.dataset_path),
           in_channels=dataset_class.get_in_channels(),
           k=args.k_folds, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
           device=device, save_models=True, early_stop=args.early_stop,
           n_streams=args.n_streams, n_streams_depth=args.n_streams_depth,
           starting_conv_kernel_size=args.starting_conv_kernel_size, n_classification_depth=args.n_classification_depth,
           logs_dir=logs_dir, logging_interval=args.epochs // 20,
           use_batchnorm=args.use_batchnorm, learned_pooling=args.learned_pooling)
elif args.experiment_type == "within_subject":
    for s_id in tqdm(dataset_class.get_subject_ids(),
                     desc=f"Doing {args.k_folds}-fold cross validation within-subject"):
        k_fold(dataset=dataset_class(data_path=args.dataset_path, subjects_to_include=s_id),
               checkpoint_path=join("weights", "bcinet.ckpt"),
               in_channels=dataset_class.get_in_channels(),
               k=args.k_folds, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
               device=device, save_models=False, early_stop=args.early_stop,
               logs_dir=join(logs_dir, f"subject_{s_id}"), logging_interval=0,
               verbose=False)

print(f"Training finished at {datetime.now().time().replace(microsecond=0)} "
      f"after {int((time.time() - starting_time) // 60)} minutes")
