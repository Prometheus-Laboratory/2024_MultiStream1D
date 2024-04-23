from collections import OrderedDict
from pprint import pprint
from typing import Optional

import numpy as np
import pandas as pd

from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics


class BCINet(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 adaptive_pooling_output_size: int = 32,
                 learning_rate: float = 1e-4,
                 dropout_probability: float = 0.5,
                 use_stream1: bool = True, use_stream2: bool = True, use_stream3: bool = True,
                 classification_blocks: int = 3,
                 device: Optional[str] = None):
        super(BCINet, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            assert device in {"cpu", "cuda"}
        self.learning_rate = learning_rate

        assert use_stream1 or use_stream2 or use_stream3
        self.use_stream1, self.use_stream2, self.use_stream3 = use_stream1, \
                                                               use_stream2, \
                                                               use_stream3

        assert classification_blocks >= 1
        self.classification_blocks = classification_blocks

        # initializes the streams
        self.streams = nn.ModuleList()
        if self.use_stream1:
            self.streams.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
                    nn.ReLU(),
                )
            )

        if self.use_stream2:
            self.streams.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=5),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU()
                )
            )

        if self.use_stream3:
            self.streams.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=256, out_channels=128, kernel_size=7),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            )

        self.reshape_layer = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=adaptive_pooling_output_size),
            nn.Flatten(start_dim=1),
        )

        self.classifier = []
        classification_start_features = 64 * adaptive_pooling_output_size * len(self.streams)
        classification_end_features = 2

        features_space = np.linspace(start=classification_start_features, stop=classification_end_features,
                                     num=self.classification_blocks + 1,
                                     dtype=int, endpoint=True)
        for i_classification_block in range(self.classification_blocks):
            in_features, out_features = None, None
            if i_classification_block == 0:
                in_features = classification_start_features
            if i_classification_block == self.classification_blocks - 1:
                out_features = classification_end_features

            if not in_features:
                in_features = features_space[i_classification_block]
            if not out_features:
                out_features = features_space[i_classification_block + 1]

            if i_classification_block < self.classification_blocks - 1:
                self.classifier.append(
                    nn.Sequential(OrderedDict([
                        ("dropout", nn.Dropout(p=dropout_probability)),
                        ("linear", nn.Linear(in_features=in_features, out_features=out_features)),
                        ("activation", nn.ReLU())
                    ]))
                )
            else:
                self.classifier.append(
                    nn.Sequential(OrderedDict([
                        ("dropout", nn.Dropout(p=dropout_probability)),
                        ("linear", nn.Linear(in_features=in_features, out_features=out_features)),
                    ]))
                )
        self.classifier = nn.Sequential(*self.classifier)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout_probability),
        #     nn.Linear(64 * adaptive_pooling_output_size * len(self.streams), 4096),
        #     nn.ReLU(),
        #
        #     nn.Dropout(p=dropout_probability),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #
        #     nn.Dropout(p=dropout_probability),
        #     nn.Linear(4096, 2)
        # )

        self._stats = []

        self.to(device)
        self.float()

    def forward(self, x):
        xs_after_streams = []
        for stream in self.streams:
            xs_after_streams += [stream(x)]
        # if self.use_stream1:
        #     xs_after_streams += [self.stream1(x)]
        # if self.use_stream2:
        #     xs_after_streams += [self.stream2(x)]
        # if self.use_stream3:
        #     xs_after_streams += [self.stream3(x)]
        for i in range(len(xs_after_streams)):
            xs_after_streams[i] = self.reshape_layer(xs_after_streams[i])

        x_after_streams = torch.cat(xs_after_streams, dim=-1)
        output = self.classifier(x_after_streams)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)
        loss = F.cross_entropy(input=y_pred, target=y)
        self.log("train_loss", loss)
        return {
            "loss": loss,
            "y": y.detach(),
            "y_pred": F.softmax(y_pred, dim=-1).detach()
        }

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        loss = F.cross_entropy(input=y_pred, target=y)
        self.log("val_loss", loss)
        return {
            "loss": loss,
            "y": y.detach(),
            "y_pred": F.softmax(y_pred, dim=-1).detach()
        }

    def training_epoch_end(self, train_outs):
        losses = torch.as_tensor([s["loss"].item() for s in train_outs])
        y = torch.cat([s["y"] for s in train_outs])
        y_pred = torch.cat([s["y_pred"] for s in train_outs])
        self._stats += [{
            "phase": "training",
            "epoch": self.current_epoch,
            "loss": torch.mean(losses).item(),
            "accuracy": torchmetrics.functional.accuracy(preds=y_pred, target=y).item(),
            "precision": torchmetrics.functional.precision(preds=y_pred, target=y).item(),
            "recall": torchmetrics.functional.recall(preds=y_pred, target=y).item(),
            "f1": torchmetrics.functional.f1_score(preds=y_pred, target=y).item(),
        }]
        self.log("train_acc", self._stats[-1]["accuracy"])
        self.log("train_prec", self._stats[-1]["precision"])
        self.log("train_rec", self._stats[-1]["recall"])
        self.log("train_f1", self._stats[-1]["f1"])

    def validation_epoch_end(self, val_outs):
        losses = torch.as_tensor([s["loss"].item() for s in val_outs])
        y = torch.cat([s["y"] for s in val_outs])
        y_pred = torch.cat([s["y_pred"] for s in val_outs])
        self._stats += [{
            "phase": "validation",
            "epoch": self.current_epoch,
            "loss": torch.mean(losses).item(),
            "accuracy": torchmetrics.functional.accuracy(preds=y_pred, target=y).item(),
            "precision": torchmetrics.functional.precision(preds=y_pred, target=y).item(),
            "recall": torchmetrics.functional.recall(preds=y_pred, target=y).item(),
            "f1": torchmetrics.functional.f1_score(preds=y_pred, target=y).item(),
        }]
        self.log("val_acc", self._stats[-1]["accuracy"])
        self.log("val_prec", self._stats[-1]["precision"])
        self.log("val_rec", self._stats[-1]["recall"])
        self.log("val_f1", self._stats[-1]["f1"])

    def on_train_epoch_end(self):
        if self.current_epoch == 0 or self.trainer.log_every_n_steps == -1 or \
                self.current_epoch % self.trainer.log_every_n_steps != 0:
            return

        stats = self.get_stats()
        print()
        print(f"Latest epochs:")
        print(stats.iloc[-3 * 2:])

        print()
        print(f"Best epochs:")
        print(stats[stats["phase"] == "validation"].sort_values(by=["accuracy"], ascending=False).iloc[:3])
        print()

    def get_stats(self):
        stats = pd.DataFrame(data=self._stats).sort_values(by=["epoch"])
        return stats
