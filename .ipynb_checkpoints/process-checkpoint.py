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
import models as mo

from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dna

sys.path.insert(0,'/Users/k2585057/borzoi/')
from examples.borzoi_helpers import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

bz_path = '/Users/k2585057/Dropbox/PhD/Analysis/Project/GENOME_PREDICT/models/borzoi/'
gn_path = '/Users/k2585057/Dropbox/PhD/Analysis/Project/GENOME/'


#===============================================================
def onehot(fa=None, chrom=None, center_pos=None, seq_len = None):
#===============================================================
    """
    one-hot encode a genomic window centered on `center_pos`.

    Parameters
    ----------
    fa : pysam.FastaFile
        opened FASTA
    chrom : str
        chromosome name (e.g. "chr1")
    center_pos : int
        genomic coordinate (0-based)
    seq_len : int
        total window size (bp)

    Returns
    -------
    np.ndarray
        1×seq_len×4 one-hot encoded sequence
    """
    
    #search_gene = 'ENSG00000144285.23' #ADD IN LATER! 
    start = center_pos - (seq_len // 2) #start of 1hot seq
    end = center_pos + (seq_len // 2) #end of 1hot seq
    
    #1hot encode
    wt_code  = process_sequence(fa, chrom, start, end) #1hot encode
    return(wt_code, start)

#========================================
def mutate(wt_code, poses, start, alts):
#========================================
    """
    Apply one or more SNVs to a one-hot encoded sequence.

    Parameters
    ----------
    wt_code : np.ndarray
        one-hot encoded sequence (shape: [seq_len, 4])
    poses : list[int]
        genomic coordinates of mutations (absolute coords)
    start : int
        genomic coordinate of the first base in wt_code
        (used to convert absolute coords → local indices)
    alts : list[str]
        alt alleles (A,C,G,T) in corresponding order to poses

    Returns
    -------
    np.ndarray
        mutated one-hot encoded sequence
    """

    #Induce mutation(s)
    mut_code = np.copy(wt_code)
    for pos, alt in zip(poses, alts) :
        alt_ix = -1
        if alt == 'A' :
            alt_ix = 0
        elif alt == 'C' :
            alt_ix = 1
        elif alt == 'G' :
            alt_ix = 2
        elif alt == 'T' :
            alt_ix = 3
    
        mut_code[pos-start-1] = 0.
        mut_code[pos-start-1, alt_ix] = 1.
    return(mut_code)