#imports
#GPU

import sys, os
print("Python:", sys.executable)
print("LD_LIBRARY_PATH:", repr(os.environ.get("LD_LIBRARY_PATH")))
print("CUDA_HOME:", repr(os.environ.get("CUDA_HOME")))
print("CUDA_PATH:", repr(os.environ.get("CUDA_PATH")))

import jax
print("JAX:", jax.__version__)
print("Devices:", jax.devices())

import os
from pathlib import Path
fdata='/home/dburrows/DATA/'


hf_home = Path(f"{fdata}/GENE_PREDICT/models/alphagenome/hf_home")
hf_home.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(hf_home)
# optional, but makes it explicit:
os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")

from alphagenome_research.model import dna_model
import numpy as np
import pandas as pd
import os

import jax
import alphagenome
import alphagenome_research
from alphagenome.data import genome
from alphagenome.models import dna_client
import time
import pysam
from tqdm import tqdm
import itertools




#find locations, regions
#=========================
loc_df=pd.read_csv(f'{fdata}/GENOME/scn1a/scn1a_aaron/4_SCN1A_variants_borzoi.bed', sep='\t').iloc[:,0].reset_index()
scn1a_df=pd.read_csv(f'{fdata}/GENOME/scn1a/scn1a_aaron/1_SCN1A_gene.bed', sep='\t', header=None)
b1_df = pd.read_csv(f'{fdata}/GENE_PREDICT/importance_score/b1_20bp_single-ism_AG.csv', index_col=0)
b2_df = pd.read_csv(f'{fdata}/GENE_PREDICT/importance_score/b2_20bp_single-ism_AG.csv', index_col=0)
a1_df = pd.read_csv(f'{fdata}/GENE_PREDICT/importance_score/a1_20bp_single-ism_AG.csv', index_col=0)

#Load genome seq
#=================
fasta = pysam.FastaFile(f"{fdata}/GENOME/annotations/hg38/assembly/ucsc/hg38.fa")


#load models
#=========================
from alphagenome_research.model import dna_model
model = dna_model.create_from_huggingface("all_folds")
fold0 = dna_model.create_from_huggingface("fold_0")
fold1 = dna_model.create_from_huggingface("fold_1")
fold2 = dna_model.create_from_huggingface("fold_2")
fold3 = dna_model.create_from_huggingface("fold_3")
model_l = [fold0, fold1, fold2, fold3]
print("Models loaded.")


# mean expr
#===============
def mean_reg(model=None, seq=None, chro=None, interval=None, brain_terms=None):
    vals = []
    for region_name, term in brain_terms.items():
        pred = model.predict_sequence(
            sequence=seq,
            requested_outputs=[dna_model.OutputType.RNA_SEQ],
            ontology_terms=[term],
            interval=interval,
        )
        vals.append(np.mean(pred.rna_seq.values, axis=1))
    return np.mean(np.array(vals), axis=0)


#expression across models
def expr_map(all_model = None, model_l=None, seq=None, chro=None, interval=None, brain_terms=None):
    point_est = mean_reg(model=all_model, seq=seq, chro=chro, interval=interval, brain_terms=brain_terms)
    fold_l = list(range(len(model_l)))
    for v,mo in enumerate(model_l):
        fold_l[v] = mean_reg(model=mo, seq=seq, chro=chro, interval=interval, brain_terms=brain_terms)
    return(point_est, np.array(fold_l))

#mutate arbitrary sequence
def mutate_seq(seq, pos, base):
    base = base.upper()
    if base not in {"A", "C", "G", "T"}:
        raise ValueError("base must be one of A, C, G, T")
    if pos < 0 or pos >= len(seq):
        raise IndexError("pos out of range")
    if seq[pos].upper() == base:
        raise ValueError("base is already the same at this position")

    seq = list(seq)
    seq[pos] = base
    return "".join(seq)

# mask sites to mutate over
def mask(mode=None, neg_thr=-0.03, neu_thr=(-0.03,0.05), reg=None, curr_df=None):
    mins = curr_df.groupby('pos')['logFC'].min()

    if mode == 'neg':
        keep_pos = mins[mins < neg_thr].index
    elif mode == 'neu':
        keep_pos = mins[(mins > neu_thr[0]) & (mins < neu_thr[1])].index
    else:
        print('mode must equal neg or neu')
        return None

    return curr_df[curr_df['pos'].isin(keep_pos)].set_index('pos')

#get sequence window
def process(loc_df=None, scn1a_df = None, fasta = None, win_len=1048576, reg=None):
    curr_df = loc_df[loc_df['level_3'] == reg].copy()
    
    chro = str(curr_df['level_0'].iloc[0]) 
    region_start = int(curr_df['level_1'].iloc[0])
    region_end = int(curr_df['level_2'].iloc[0])
    cntr = (region_start + region_end) // 2
    
    start = cntr - (win_len//2)
    end = cntr + (win_len//2)
    interval = genome.Interval(chro, start, end)
    
    #mutation end and start relative to window
    region_rel_start = (region_start - start)
    region_rel_end = (region_end - start)
    
    #promoter end and start (for gene expression)
    gene_start = scn1a_df[1].values[0]
    gene_end = scn1a_df[2].values[0]
    gene_rel_start = (gene_start - start)
    gene_rel_end = (gene_end-start)
    
    
    print(interval)
    print("Width:", interval.width)

    seq = fasta.fetch(chro, start, end)
    return(chro, region_start, 
           region_end, cntr, interval, 
           region_rel_start, region_rel_end,
           gene_start, gene_end, gene_rel_start,
           gene_rel_end, seq)


def _run(reg, mode, curr_df):
    #models
    model_l = [fold0, fold1, fold2, fold3]
    
    # Brain-associated ontology terms 
    brain_terms = {
        "brain": "UBERON:0000955",
        "frontal_cortex": "UBERON:0001870",
        # "caudate_nucleus": "UBERON:0001873",
        # "putamen": "UBERON:0001874",
        # "amygdala": "UBERON:0001876",
        # "nucleus_accumbens": "UBERON:0001882",
        # "hypothalamus": "UBERON:0001898",
        "hippocampus_ammons_horn": "UBERON:0001954",
        # "cerebellum": "UBERON:0002037",
        # "substantia_nigra": "UBERON:0002038",
        # "cerebellar_hemisphere": "UBERON:0002245",
        # "spinal_cord_c1": "UBERON:0006469",
        "dlpfc_ba9": "UBERON:0009834",
        "anterior_cingulate_ba24": "UBERON:0009835",
    }
    
    # Extract sequence
    #===================
    (chro, region_start, region_end, cntr, interval, 
    region_rel_start, region_rel_end,
    gene_start, gene_end, gene_rel_start,
    gene_rel_end, seq 
    )= process(loc_df=loc_df, scn1a_df = scn1a_df, fasta = fasta, win_len=1048576, reg = reg)
    
    #generate baseline
    #======================
    baseline,_ = expr_map(all_model=model, model_l = model_l, seq=seq, chro=chro, interval=interval, brain_terms=brain_terms)
    base_mean = np.mean(baseline[gene_rel_start:gene_rel_end])
    
    # #Run ISM
    # #=================
    #Define mask
    neg_thr = -0.03
    neu_thr = (-0.03,0.05)
    mask_df = mask(mode = mode, neg_thr = neg_thr, neu_thr = neu_thr,
                   reg=reg, curr_df = curr_df)
    mask_rel_pos = np.array(mask_df.index.unique())
    
    
    # allowed alts per position from mask_df itself
    alt_by_pos = []
    ref_by_pos = []
    for pos in mask_rel_pos:
        sub = mask_df.loc[pos]
        sub = sub.sort_values('alt')
        ref_by_pos.append(sub['ref'].iloc[0])
        alt_by_pos.append(sub['alt'].tolist())
    
    print('n positions:', len(mask_rel_pos))
    print('total combos:', np.prod([len(x) for x in alt_by_pos]))
    
    # Combinatorial ISM
    # =================
    rows = []
    
    for combo in tqdm(itertools.product(*alt_by_pos), total=int(np.prod([len(x) for x in alt_by_pos]))):
        
        # mutate all selected positions at once
        mut_seq = list(seq)
        for pos, alt in zip(mask_rel_pos, combo):
            mut_seq[pos] = alt
        mut_seq = ''.join(mut_seq)
        
        # predict
        point, fold_outs = expr_map(
            all_model=model,
            model_l=model_l,
            seq=mut_seq,
            chro=chro,
            interval=interval,
            brain_terms=brain_terms
        )
        
        mut_mean = np.mean(point[gene_rel_start:gene_rel_end])
        logFC = np.log2((mut_mean + 1e-8) / (base_mean + 1e-8))
        sd = np.mean(np.std(fold_outs[:,gene_rel_start:gene_rel_end],axis=0))
        
        rows.append({
            'reg': reg,
            'mode': mode,
            'positions': tuple(mask_rel_pos),
            'refs': ''.join(ref_by_pos),
            'alts': ''.join(combo),
            'n_mut': len(combo),
            'mut_mean': mut_mean,
            'base_mean': base_mean,
            'delta': mut_mean - base_mean,
            'logFC': logFC,
            'sd': sd
        })
    
    comb_df = pd.DataFrame(rows).sort_values('logFC')
    comb_df.to_csv(f'{fdata}/GENE_PREDICT/ISM/{reg}_{mode}_20bp_alphagenome.csv')



reg_l = ['B2', 'B1', 'A1']
df_l = [b2_df, b1_df, a1_df]
mode_l = ['neg', 'neu']

for g,reg in enumerate(reg_l):
    curr_df = df_l[g]
    for mode in mode_l: 
        _run(reg,mode, curr_df)
        print(f'Done {reg} for {mode}') 