from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd

from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics


class MultiStream1dNet(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 adaptive_pooling_output_size: int = 32,
                 learning_rate: float = 1e-4,
                 dropout_probability: float = 0.5,
                 n_streams: int = 4,
                 n_streams_depth: int = 1,
                 n_classification_depth: int = 3,
                 starting_conv_kernel_size: int = 3,
                 use_batchnorm: bool = False,
                 learned_pooling: bool = False,
                 max_channels: int = 256,
                 device: Optional[str] = None):
        super(MultiStream1dNet, self).__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            assert device in {"cpu", "cuda"}
        self.learning_rate = learning_rate

        assert in_channels >= 1
        self.in_channels = in_channels

        assert max_channels >= 1
        self.max_channels = max_channels

        assert n_streams >= 1
        self.n_streams = n_streams

        assert n_streams_depth >= 1
        self.n_streams_blocks = n_streams_depth

        assert n_classification_depth >= 1
        self.classification_blocks = n_classification_depth

        assert starting_conv_kernel_size >= 1
        self.starting_conv_kernel_size = starting_conv_kernel_size

        # initializes the streams
        self.streams = nn.ModuleList()
        for i_stream in range(self.n_streams):
            stream = []
            kernel_size = self.starting_conv_kernel_size + i_stream * 2
            channels_up = np.geomspace(start=self.in_channels, stop=self.max_channels, num=self.n_streams_blocks,
                                       endpoint=True, dtype=int)
            channels_down = np.geomspace(start=self.max_channels, stop=64, num=self.n_streams_blocks + 1,
                                         endpoint=True, dtype=int)
            channels = np.concatenate([channels_up, channels_down])
            for i_stream_block in range(self.n_streams_blocks):
                # main blocks
                if i_stream_block < self.n_streams_blocks - 1:
                    stream += [self._make_conv_block(kernel_size=kernel_size,
                                                     in_channels=channels[i_stream_block * 2],
                                                     mid_channels=channels[i_stream_block * 2 + 1],
                                                     out_channels=channels[i_stream_block * 2 + 2],
                                                     use_batchnorm=use_batchnorm,
                                                     use_pooling=True, learned_pooling=learned_pooling)]
                # last block
                else:
                    stream += [self._make_conv_block(kernel_size=kernel_size,
                                                     in_channels=channels[i_stream_block * 2],
                                                     mid_channels=channels[i_stream_block * 2 + 1],
                                                     out_channels=channels[i_stream_block * 2 + 2],
                                                     use_batchnorm=use_batchnorm,
                                                     use_pooling=False, learned_pooling=learned_pooling)]
            self.streams.append(nn.Sequential(*stream))

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

        self._stats = []

        self.to(device)
        self.float()

    def _make_conv_block(self, kernel_size: int,
                         in_channels: int, mid_channels: int, out_channels: int,
                         use_batchnorm: bool = False,
                         use_pooling: bool = True, learned_pooling: bool = True):
        conv_block = []
        for i in range(2):
            # adds a conv
            conv_block += [
                (f"conv_{i + 1}", nn.Conv1d(in_channels=in_channels if i == 0 else mid_channels,
                                            out_channels=mid_channels if i == 0 else out_channels,
                                            padding=kernel_size // 2,
                                            kernel_size=(kernel_size,)),),
            ]
            # eventually adds batch norm
            if use_batchnorm:
                conv_block += [
                    (f"bn_{i + 1}", nn.BatchNorm1d(num_features=mid_channels if i == 0 else out_channels))
                ]
            # adds the activation function
            conv_block += [
                (f"activation_{i + 1}", nn.ReLU())
            ]
        # eventually adds pooling
        if use_pooling:
            if not learned_pooling:
                conv_block += [
                    (f"pooling", nn.MaxPool1d(kernel_size=kernel_size,
                                              stride=2,
                                              padding=kernel_size // 2))
                ]
            else:
                conv_block += [
                    (f"pooling", nn.Conv1d(in_channels=out_channels,
                                           out_channels=out_channels,
                                           stride=2,
                                           padding=kernel_size // 2,
                                           kernel_size=(kernel_size,)))
                ]
        return nn.Sequential(OrderedDict(conv_block))

    def forward(self, x):
        xs_after_streams = []
        for stream in self.streams:
            xs_after_streams += [stream(x)]

        for i in range(len(xs_after_streams)):
            xs_after_streams[i] = self.reshape_layer(xs_after_streams[i])

        x_after_streams = torch.cat(xs_after_streams, dim=-1)

        output = self.classifier(x_after_streams)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)
        loss = F.cross_entropy(input=y_pred, target=y)
        self.log("train_loss", loss, prog_bar=True)
        return {
            "loss": loss,
            "y": y.detach(),
            "y_pred": F.softmax(y_pred, dim=-1).detach()
        }

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        loss = F.cross_entropy(input=y_pred, target=y)
        self.log("val_loss", loss, prog_bar=True)
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
        self.log("train_acc", self._stats[-1]["accuracy"], prog_bar=True)
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
        self.log("val_acc", self._stats[-1]["accuracy"], prog_bar=True)
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
