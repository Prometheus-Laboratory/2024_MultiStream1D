# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 23:05:19 2021

@author: odraode
"""

'''
Salvataggio del RAW data

'''




import mne

import numpy as np

from mne.io import concatenate_raws

#prendo quei soggetti che hanno dei dati consoni con cui lavorare (scartati 43, 88, 89, 92, 100 , 104)
indici=[]
for x in range(1,43):
    indici.append(x)

for x in range(44,88):
    indici.append(x)
    
for x in range(90,92):
    indici.append(x)
    
for x in range(93,100):
    indici.append(x)

for x in range(101,104):
    indici.append(x)

for x in range(105,108):
    indici.append(x)



#ricavo i dati da tutti i soggetti
physionet_paths = [ mne.datasets.eegbci.load_data(sub_id,[3,4,7,8,11,12,],path='eegbci/' , update_path=False) for sub_id in indici]

#lettura dei dati
physionet_paths = np.concatenate(physionet_paths)
parts = [mne.io.read_raw_edf(path, preload=True,stim_channel='auto')
         for path in physionet_paths]

#concateno i dati da tutti i soggetti per ottenerne uno solo su cui lavorarci
raw = concatenate_raws(parts)
del parts

raw.save("final_raw_eeg.fif")



























