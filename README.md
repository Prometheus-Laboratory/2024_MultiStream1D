# Repository of the code for the paper entitled ''Multi-Stream 1D CNN for EEG Motor Imagery Classification of Limbs Activation'' 

## Training on Physionet Motor Imagery
The structure of the dataset's folder must contain the following folders and files, downloadable from [the official webpage](https://physionet.org/content/eegmmidb/1.0.0/):

```
S001
S002
...
S109
wfdblcal
```

To run the training, run the following:

`python training.py physionet <path to dataset>`