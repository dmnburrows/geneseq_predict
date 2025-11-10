import json
import os
import sys
import time
import warnings
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pysam
import pyfaidx 
import pyranges as pr
import tensorflow as tf

from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dna

sys.path.insert(0,'/Users/k2585057/borzoi/')
from examples.borzoi_helpers import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



#==================================================
def borzoi_load(n_folds=4, seq_len=524288, rc=True):
#==================================================
    """
    Load a Borzoi ensemble.
    
    Returns a list of `SeqNN` models restored from disk, sliced to the GTEx
    target set, and optionally configured for reverse-complement ensembling.
    
    Parameters
    ----------
    n_folds : int
        number of folds/models to load
    seq_len : int
        model sequence length (default ~520kb; not modified here)
    rc : bool
        enable reverse-complement ensembling
    
    Returns
    -------
    list of seqnn.SeqNN
        loaded models
    """

    bz_path = '/Users/k2585057/Dropbox/PhD/Analysis/Project/GENOME_PREDICT/models/borzoi/'
    params_file = f'{bz_path}/examples/params_pred.json'
    targets_file = f'{bz_path}/examples/targets_gtex.txt' #Subset of targets_human.txt
    
    #Read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
        params_model = params['model']
        params_train = params['train']
    
    #Read targets
    targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
    target_index = targets_df.index
    assert all(targets_df['strand_pair'].values == targets_df.index), 'strand pairs dont match indeces - may causes errors later'

    
    #Create local index of strand_pair (relative to sliced targets)
    #THIS SEEMS WEIRD - MAYBE FIX LATER!
    if rc :
        strand_pair = targets_df.strand_pair 
        
        target_slice_dict = {ix : i for i, ix in enumerate(target_index.values.tolist())}
        slice_pair = np.array([
            target_slice_dict[ix] if ix in target_slice_dict else ix for ix in strand_pair.values.tolist()
        ], dtype='int32')
    
    #Initialize model ensemble
    #==========================
    models = []
    for fold_ix in range(n_folds) :
        model_file = f'/Users/k2585057/Dropbox/PhD/Analysis/Project/GENOME_PREDICT/models/borzoi/saved_models/f3c{str(fold_ix)}/train/model0_best.h5'
        seqnn_model = seqnn.SeqNN(params_model)
        seqnn_model.restore(model_file, 0)
        seqnn_model.build_slice(target_index)
        if rc :
            seqnn_model.strand_pair.append(slice_pair)
        seqnn_model.build_ensemble(rc, [0])
        models.append(seqnn_model)
    return(models)
    