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

bz_path = '/Users/k2585057/borzoi/'
gn_path = '/Users/k2585057/Dropbox/PhD/Analysis/Project/GENOME/'


import numpy as np
import pandas as pd

def process(models, wt_code, mut_code,
                                      window_start, window_end,
                                      gene_start, gene_end,
                                      channel_indices,
                                      channel_aggregate='mean',
                                      bin_normalize=False,
                                      reduce='sum',
                                      pseudocount=1e-6):
    """
    Returns:
      df: tidy per-fold dataframe (WT & MUT gene-level expr, delta, log2FC)
      tracks: dict with:
        x            (Nbins,)
        wt_tracks    (F, Nbins)
        mut_tracks   (F, Nbins)
        resid_tracks (F, Nbins)  = mut - wt
        wt_mean, wt_sd           (Nbins,)
        mut_mean, mut_sd         (Nbins,)
        resid_mean, resid_sd     (Nbins,)
    Notes:
      - bin_normalize=False recommended for expression deltas/log2FC.
      - region used for tracks is the gene span [min(gene_start, gene_end), max(...)].
    """
    # region (ascending coords)
    region_start = min(gene_start, gene_end)
    region_end   = max(gene_start, gene_end)

    # to collect fold-level tracks & scalars
    wt_tracks, mut_tracks, resid_tracks = [], [], []
    records = []

    x_ref = None  # will capture bin centers once

    for fold_ix, mdl in enumerate(models):
        # per-fold predictions (IMPORTANT: wrap single model)
        y_wt_f  = predict_tracks([mdl], wt_code)  # (1,1,B,T)
        y_mut_f = predict_tracks([mdl], mut_code)

        # --- aggregate channels & slice region (returns x_reg, t_reg) ---
        x_wt, t_wt = aggregate_region(y_wt_f,  window_start, window_end,
                                      region_start, region_end,
                                      channel_indices, aggregate=channel_aggregate,
                                      normalize_counts=bin_normalize)
        x_mut, t_mut = aggregate_region(y_mut_f, window_start, window_end,
                                        region_start, region_end,
                                        channel_indices, aggregate=channel_aggregate,
                                        normalize_counts=bin_normalize)

        if x_ref is None:
            x_ref = x_wt
        else:
            # sanity: region binning should match across folds
            assert np.allclose(x_ref, x_wt), "Inconsistent x bins across folds"

        # gene-level scalars
        expr_wt  = gene_expression_from_bins(x_wt,  t_wt,  gene_start, gene_end, reduce=reduce)
        expr_mut = gene_expression_from_bins(x_mut, t_mut, gene_start, gene_end, reduce=reduce)
        delta    = expr_mut - expr_wt
        log2fc   = np.log2((expr_mut + pseudocount) / (expr_wt + pseudocount))

        # store tracks
        wt_tracks.append(t_wt)
        mut_tracks.append(t_mut)
        resid_tracks.append(t_mut - t_wt)

        # store rows
        records.append({'fold': fold_ix, 'condition': 'WT',
                        'expr': expr_wt, 'delta_vs_WT': 0.0, 'log2FC_vs_WT': 0.0})
        records.append({'fold': fold_ix, 'condition': 'MUT',
                        'expr': expr_mut, 'delta_vs_WT': delta, 'log2FC_vs_WT': log2fc})

    # stack tracks: (F, Nbins)
    wt_arr    = np.stack(wt_tracks, axis=0)
    mut_arr   = np.stack(mut_tracks, axis=0)
    resid_arr = np.stack(resid_tracks, axis=0)

    # fold means / sds (per bin)
    wt_mean, wt_sd         = wt_arr.mean(0), wt_arr.std(0)
    mut_mean, mut_sd       = mut_arr.mean(0), mut_arr.std(0)
    resid_mean, resid_sd   = resid_arr.mean(0), resid_arr.std(0)

    # assemble dataframe
    df = pd.DataFrame.from_records(records)

    tracks = dict(
        x=x_ref,
        wt_tracks=wt_arr, mut_tracks=mut_arr, resid_tracks=resid_arr,
        wt_mean=wt_mean, wt_sd=wt_sd,
        mut_mean=mut_mean, mut_sd=mut_sd,
        resid_mean=resid_mean, resid_sd=resid_sd
    )
    return df, tracks

def mutate(wt_code, poses, alts):

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

import matplotlib.pyplot as plt
import seaborn as sns

def plot_foldwise_expression(df, title="SCN1A predicted expression (brain tracks)"):
    """
    df: output from foldwise_expression_df
    """
    # Pivot for convenience: fold × condition → expr
    expr_wide = df.pivot(index='fold', columns='condition', values='expr')
    fold_means = expr_wide.mean()
    fold_sds   = expr_wide.std()

    plt.figure(figsize=(6,4))
    
    # plot per fold as paired dots + lines
    for fold, row in expr_wide.iterrows():
        plt.plot(['WT','MUT'], row.values, marker='o', linestyle='-', alpha=0.7, color='gray')

    # overlay mean ± sd
    plt.errorbar(['WT','MUT'], fold_means.values, yerr=fold_sds.values,
                 fmt='o', markersize=10, color='black', capsize=5, label='Mean ± SD')

    plt.ylabel("Predicted expression (sum of bins)")
    plt.title(title)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_foldwise_delta(df, title="Change in expression per fold"):
    """Plot bar chart of per-fold Δ and log2FC."""
    mut_rows = df[df['condition']=='MUT'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    sns.barplot(x='fold', y='delta_vs_WT', data=mut_rows, ax=axes[0], color='tab:blue')
    axes[0].set_title("Δ (MUT − WT)")
    axes[0].axhline(0, color='k', linewidth=1)

    sns.barplot(x='fold', y='log2FC_vs_WT', data=mut_rows, ax=axes[1], color='tab:orange')
    axes[1].set_title("log2 Fold Change")
    axes[1].axhline(0, color='k', linewidth=1)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()



#load annotations
#===================================
#Initialize fasta sequence extractor 
fasta_open = pysam.Fastafile(f'{gn_path}/annotations/hg38/assembly/ucsc/hg38.fa')
#Load GTF (optional; needed to compute exon coverage attributions for example gene)
transcriptome = bgene.Transcriptome(f'{gn_path}/annotations/hg38/genes/gencode41/gencode41_basic_nort.gtf')

#Model configuration
#=============================
params_file = f'{bz_path}/examples/params_pred.json'
targets_file = f'{bz_path}/examples/targets_gtex.txt' #Subset of targets_human.txt
seq_len = 524288
rc = True         #Average across reverse-complement prediction
n_folds = 4       #To use only one model fold, set to 'n_folds = 1'. To use all four folds, set 'n_folds = 4'.


#SCN1A promoter positions
#=============================
P1a= [166148180, 166151550]
P1b= [166127360, 166129030]
P1c= [166077140, 166079490]
scn1a = [165984641, 166182806]


#================================
#load model params 
#===================================

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
    model_file = f'/Users/k2585057/Dropbox/PhD/Analysis/Project/SCN1A_PREDICT/BORZOI/saved_models/f3c{str(fold_ix)}/train/model0_best.h5'
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, 0)
    seqnn_model.build_slice(target_index)
    if rc :
        seqnn_model.strand_pair.append(slice_pair)
    seqnn_model.build_ensemble(rc, [0])
    
    models.append(seqnn_model)



import re

# ====== MINIMAL BATCH: LOOP → PREDICT → SAVE ======

# config you already have
seq_len   = 524_288
brain_idx = [17, 18, 19]   # average across these channels
out_dir   = "borzoi_out_min"
os.makedirs(out_dir, exist_ok=True)

# SCN1A coords and promoters (minus strand → use lower coord as TSS-ish)
SCN1A = [165_984_641, 166_182_806]
P1a = [166_148_180, 166_151_550]
P1b = [166_127_360, 166_129_030]
P1c = [166_077_140, 166_079_490]
prom_TSS = {'P1a': P1a[0], 'P1b': P1b[0], 'P1c': P1c[0]}

# your variant lists
prom_l = ['P1a', 'P1a', 'P1c', 'P1b', 'P1a', 'P1a']
mut_l  = ['chr2-166149776-A-T', 'chr2-166150870-T-C', 'chr2-166077919-T-A',
          'chr2-166128263-G-T', 'chr2-166148245-C-G', 'chr2-166148581-A-C']

def sanitize(v):
    return re.sub(r'[^A-Za-z0-9_\-\.]', '_', v)

def brain_aggregate(y, idxs):
    """y shape (1,1,B,T) -> (B,) mean over idxs"""
    return y[0, 0, :, idxs].mean(axis=1)

for prom_name, var in zip(prom_l, mut_l):
    chrom, pos_s, ref, alt = var.split('-')
    pos = int(pos_s)

    centers = {
        prom_name: prom_TSS[prom_name],
        'MID': (SCN1A[0] + SCN1A[1]) // 2
    }

    vdir = os.path.join(out_dir, sanitize(var))
    os.makedirs(vdir, exist_ok=True)

    for center_label, center_pos in centers.items():
        start = center_pos - (seq_len // 2)
        end   = center_pos + (seq_len // 2)

        # encode WT and apply SNP (uses your mutate(wt_code, poses, alts) that depends on 'start')
        wt_code = process_sequence(fasta_open, chrom, start, end)
        mut_code = mutate(wt_code, [pos], [alt])

        # per-fold predictions and saves
        cdir = os.path.join(vdir, center_label)
        os.makedirs(cdir, exist_ok=True)

        for fold_ix, mdl in enumerate(models):
            y_wt  = predict_tracks([mdl], wt_code)   # (1,1,B,T)
            y_mut = predict_tracks([mdl], mut_code)  # (1,1,B,T)

            # aggregate brain channels → (B,)
            wt_track  = brain_aggregate(y_wt, brain_idx)
            mut_track = brain_aggregate(y_mut, brain_idx)
            resid     = mut_track - wt_track

            # genomic bin centers
            B = y_wt.shape[2]
            bp_per_bin = (end - start) / float(B)
            x = start + bp_per_bin * (np.arange(B) + 0.5)

            # save compact npz
            np.savez(
                os.path.join(cdir, f"fold{fold_ix}.npz"),
                x=x, wt=wt_track, mut=mut_track, resid=resid,
                chrom=chrom, start=start, end=end,
                center_label=center_label, center_pos=center_pos,
                variant=var, promoter=prom_name
            )

        print(f"Saved {var} @ {center_label} → {cdir}")

print("Done. Compact per-fold results in:", out_dir)
