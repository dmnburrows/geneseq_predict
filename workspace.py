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
import tensorflow as tf

from baskerville import seqnn
from baskerville import gene as bgene
from baskerville import dna

sys.path.insert(0,'/Users/k2585057/borzoi/')
from examples.borzoi_helpers import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

bz_path = '/Users/k2585057/borzoi/'
gn_path = '/Users/k2585057/Dropbox/PhD/Analysis/Project/GENOME/annotations/'


#load annotations
#===================================

#Initialize fasta sequence extractor
fasta_open = pysam.Fastafile(f'{gn_path}/hg38/assembly/ucsc/hg38.fa')

#Load splice site annotation
splice_df = pd.read_csv(f'{gn_path}/hg38/genes/gencode41/gencode41_basic_protein_splice.csv.gz', sep='\t', compression='gzip')
print("len(splice_df) = " + str(len(splice_df)))


#load model params 
#===================================

#Model configuration
params_file = f'{bz_path}/examples/params_pred.json'
targets_file = f'{bz_path}/examples/targets_gtex.txt' #Subset of targets_human.txt

seq_len = 524288
n_folds = 1       #To use only one model fold, set to 'n_folds = 1'. To use all four folds, set 'n_folds = 4'.
rc = True         #Average across reverse-complement prediction

#Read model parameters
with open(params_file) as params_open :
    
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
    model_file = f'{gn_path}/saved_models/f3c{str(fold_ix)}/train/model0_best.h5'
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, 0)
    seqnn_model.build_slice(target_index)
    if rc :
        seqnn_model.strand_pair.append(slice_pair)
    seqnn_model.build_ensemble(rc, [0])
    models.append(seqnn_model)




#Load GTF (optional; needed to compute exon coverage attributions for example gene)
transcriptome = bgene.Transcriptome(f'{gn_path}/hg38/genes/gencode41/gencode41_basic_nort.gtf')

#NB need to ensure the gene lies within this range???
search_gene = 'ENSG00000144285.23'
center_pos = 166_060_000
chrom = 'chr2'

start = center_pos - seq_len // 2
end = center_pos + seq_len // 2

print(start)

#Get exon bin range
gene_keys = [gene_key for gene_key in transcriptome.genes.keys() if search_gene in gene_key]

print(gene_keys)

gene = transcriptome.genes[gene_keys[0]]

#Determine output sequence start
#NB dont fully understand whats going on here! 
seq_out_start = start + seqnn_model.model_strides[0]*seqnn_model.target_crops[0]
seq_out_len = seqnn_model.model_strides[0]*seqnn_model.target_lengths[0]

print(seq_out_start)
print(seq_out_len)

#Determine output positions of gene exons
gene_slice = gene.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False)


search_gene = 'ENSG00000187164'
center_pos = 116952944
chrom = 'chr10'
poses = [116952944]
alts = ['C']

start = center_pos - seq_len // 2
end = center_pos + seq_len // 2

#Get exon bin range
gene_keys = [gene_key for gene_key in transcriptome.genes.keys() if search_gene in gene_key]

gene = transcriptome.genes[gene_keys[0]]

#Determine output sequence start
seq_out_start = start + seqnn_model.model_strides[0]*seqnn_model.target_crops[0]
seq_out_len = seqnn_model.model_strides[0]*seqnn_model.target_lengths[0]

#Determine output positions of gene exons
gene_slice = gene.output_slice(seq_out_start, seq_out_len, seqnn_model.model_strides[0], False)


#Print index of GTEx blood and muscle tracks in targets file

targets_df['local_index'] = np.arange(len(targets_df))

print("blood tracks = " + str(targets_df.loc[targets_df['description'] == 'RNA:blood']['local_index'].tolist()))
print("muscle tracks = " + str(targets_df.loc[targets_df['description'] == 'RNA:muscle']['local_index'].tolist()))
print("brain tracks = " + str(targets_df.loc[targets_df['description'] == 'RNA:brain']['local_index'].tolist()))
print(list(targets_df['description']))


#make predictions
save_figs = False
save_suffix = '_chr10_116952944_T_C'
sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)

#Induce mutation(s)
sequence_one_hot_mut = np.copy(sequence_one_hot_wt)

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

    sequence_one_hot_mut[pos-start-1] = 0.
    sequence_one_hot_mut[pos-start-1, alt_ix] = 1.


#Make predictions
y_wt = predict_tracks(models, sequence_one_hot_wt)
y_mut = predict_tracks(models, sequence_one_hot_mut)