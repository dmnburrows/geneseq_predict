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

#Load genome seq
#=================
fasta = pysam.FastaFile(f"{fdata}/GENOME/annotations/hg38/assembly/ucsc/hg38.fa")

#load models
#=========================
from alphagenome_research.model import dna_model
model = dna_model.create_from_huggingface("all_folds")
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
    # fold_l = list(range(len(model_l)))
    # for v,mo in enumerate(model_l):
    #     fold_l[v] = mean_reg(model=mo, seq=seq, chro=chro, interval=interval, brain_terms=brain_terms)
    return(point_est)#, np.array(fold_l))

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


def prepare_region(loc_df=None, scn1a_df=None, fasta=None, win_len=1048576, reg=None):
    brain_terms = {
        "brain": "UBERON:0000955",
        "frontal_cortex": "UBERON:0001870",
        "hippocampus_ammons_horn": "UBERON:0001954",
        "dlpfc_ba9": "UBERON:0009834",
        "anterior_cingulate_ba24": "UBERON:0009835",
    }

    (
        chro, region_start, region_end, cntr, interval,
        region_rel_start, region_rel_end,
        gene_start, gene_end, gene_rel_start,
        gene_rel_end, seq
    ) = process(
        loc_df=loc_df,
        scn1a_df=scn1a_df,
        fasta=fasta,
        win_len=win_len,
        reg=reg,
    )

    baseline = expr_map(
        all_model=model,
        seq=seq,
        chro=chro,
        interval=interval,
        brain_terms=brain_terms,
    )

    base_mean = np.mean(baseline[gene_rel_start:gene_rel_end])

    return {
        "reg": reg,
        "brain_terms": brain_terms,

        "chro": chro,
        "region_start": int(region_start),
        "region_end": int(region_end),
        "cntr": int(cntr),
        "interval": interval,

        "region_rel_start": int(region_rel_start),
        "region_rel_end": int(region_rel_end),

        "gene_start": int(gene_start),
        "gene_end": int(gene_end),
        "gene_rel_start": int(gene_rel_start),
        "gene_rel_end": int(gene_rel_end),

        "seq": seq,
        "baseline": baseline,
        "base_mean": float(base_mean),
    }


def run_mutation(region_info, N=10):
    reg = region_info["reg"]
    brain_terms = region_info["brain_terms"]

    chro = region_info["chro"]
    interval = region_info["interval"]

    region_start = region_info["region_start"]
    region_end = region_info["region_end"]
    region_rel_start = region_info["region_rel_start"]
    region_rel_end = region_info["region_rel_end"]

    gene_rel_start = region_info["gene_rel_start"]
    gene_rel_end = region_info["gene_rel_end"]

    seq = region_info["seq"]
    base_mean = region_info["base_mean"]

    four = np.asarray(["A", "C", "G", "T"])

    start_pos = int(region_rel_start)
    end_pos = int(region_rel_end) - 1  # region_rel_end is exclusive

    if end_pos <= start_pos:
        raise ValueError("Region is too short to mutate start and end.")

    interior_positions = np.arange(start_pos + 1, end_pos)
    n_random = N - 2

    if n_random > len(interior_positions):
        raise ValueError(
            f"Need {n_random} random interior positions, "
            f"but only {len(interior_positions)} available."
        )

    random_positions = np.random.choice(
        interior_positions,
        size=n_random,
        replace=False,
    )

    # Force start + end, plus random interior positions
    its = np.concatenate([[start_pos, end_pos], random_positions]).astype(int)
    np.random.shuffle(its)

    assert len(its) == N
    assert len(its) == len(np.unique(its)), "its contains duplicate positions"
    assert start_pos in its
    assert end_pos in its

    mutseq = seq

    refs = []
    alts = []
    positions_abs_window = []
    positions_rel_region = []

    for it in its:
        it = int(it)

        curr = mutseq[it].upper()

        if curr not in {"A", "C", "G", "T"}:
            raise ValueError(f"Non-ACGT base at position {it}: {curr}")

        not_curr = four[four != curr]
        base = str(np.random.choice(not_curr))

        refs.append(curr)
        alts.append(base)
        positions_abs_window.append(it)
        positions_rel_region.append(it - region_rel_start)

        mutseq = mutate_seq(mutseq, it, base)

    # Sanity checks
    all_changed = np.array([
        i for i, (a, b) in enumerate(zip(seq.upper(), mutseq.upper()))
        if a != b
    ])

    assert len(all_changed) == N, f"Expected exactly {N} total changes, found {len(all_changed)}"
    assert set(map(int, all_changed)) == set(map(int, its)), "Changed positions do not match requested positions"
    assert seq[start_pos].upper() != mutseq[start_pos].upper(), "Start position did not mutate"
    assert seq[end_pos].upper() != mutseq[end_pos].upper(), "End position did not mutate"

    # Mutant prediction only
    point = expr_map(
        all_model=model,
        seq=mutseq,
        chro=chro,
        interval=interval,
        brain_terms=brain_terms,
    )

    mut_mean = np.mean(point[gene_rel_start:gene_rel_end])

    FC = (mut_mean + 1e-8) / (base_mean + 1e-8)
    logFC = np.log2(FC)

    return {
        "reg": reg,
        "n_mut": N,

        # mutation info
        "positions_window": tuple(map(int, positions_abs_window)),
        "positions_region": tuple(map(int, positions_rel_region)),
        "refs": "".join(refs),
        "alts": "".join(alts),

        # region info
        "chro": chro,
        "region_start": int(region_start),
        "region_end": int(region_end),
        "region_rel_start": int(region_rel_start),
        "region_rel_end": int(region_rel_end),
        "start_forced_pos": int(start_pos),
        "end_forced_pos": int(end_pos),

        # prediction info
        "base_mean": float(base_mean),
        "mut_mean": float(mut_mean),
        "delta": float(mut_mean - base_mean),
        "FC": float(FC),
        "logFC": float(logFC),
    }


# ============================================================
# Search settings
# ============================================================

reg_l = ["B2", "B1", "A1"]

N = 10

fc_low = 0.95
fc_high = 1.05

target_keep_per_reg = 20
max_attempts_per_reg = 500

mode = f"{N}mut_neutral_fc_{fc_low}_{fc_high}"

out_dir = f"{fdata}/GENE_PREDICT/ISM"
os.makedirs(out_dir, exist_ok=True)

out_path = f"{out_dir}/scn1a_{mode}_accepted.csv"

rows = []

# ============================================================
# Run search
# ============================================================

for reg in reg_l:
    kept = 0
    attempted = 0

    print(f"\n=== Preparing {reg} baseline ===")

    region_info = prepare_region(
        loc_df=loc_df,
        scn1a_df=scn1a_df,
        fasta=fasta,
        win_len=1048576,
        reg=reg,
    )

    print(f"{reg} base_mean:", region_info["base_mean"])
    print(f"\n=== Starting {reg} mutation search ===")

    pbar = tqdm(total=target_keep_per_reg, desc=f"{reg} kept")

    while kept < target_keep_per_reg and attempted < max_attempts_per_reg:
        attempted += 1

        out = run_mutation(
            region_info=region_info,
            N=N,
        )

        fc = out["FC"]

        if fc_low <= fc <= fc_high:
            kept += 1

            out["mode"] = mode
            out["attempt"] = attempted
            out["kept_idx"] = kept
            out["accepted"] = True

            rows.append(out)

            pbar.update(1)
            pbar.set_postfix({"attempts": attempted, "FC": round(fc, 4)})

            # Save after every accepted row
            keep_df = pd.DataFrame(rows)
            keep_df.to_csv(out_path, index=False)

        else:
            if attempted % 10 == 0:
                print(
                    f"Rejected {reg} attempt {attempted}: "
                    f"FC={fc:.4f}, logFC={out['logFC']:.4f}, kept={kept}"
                )

    pbar.close()

    print(
        f"Done {reg}: kept {kept}/{attempted} attempts "
        f"within FC range [{fc_low}, {fc_high}]"
    )

# ============================================================
# Final save
# ============================================================

keep_df = pd.DataFrame(rows)
keep_df.to_csv(out_path, index=False)

print(f"\nSaved accepted constructs to: {out_path}")
print(keep_df.head())
print("Total accepted:", len(keep_df))

if len(keep_df) > 0:
    print(keep_df.groupby("reg").size())
else:
    print("No accepted constructs.")