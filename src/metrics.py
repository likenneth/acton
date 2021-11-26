import os
import pickle, time
import json, math
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import entropy
from src.data.dataset.cluster_misc import lexicon, get_names, genre_list, vidn_parse
from src.data.dataset.loader import AISTDataset
from src.data.dataset.utils import save_paired_keypoints3d_as_video, rigid_align, rigid_align_sequence
from src.data.distance.nndtw import DTW

data_dir = "../aistplusplus"

def preprocess(df):
    # input: a df with cols: idx, word, length, y, name
    # split advanced dance into multiple rows or remove them
    # give each snippet a tag of corresponding base dance
    res = pd.DataFrame(columns=["idx", "word", "length", "y", "label", "name"])
    for index, row in df.iterrows():
        if "sBM" in row["name"]:
            parsed = vidn_parse(row["name"])
            tba = dict(row)
            tba["label"] = int(parsed["choreo"][2:4])
            res = res.append(tba, ignore_index=True)
        else:
            raise NotImplementedError
    return res

def metric_nmi(df):
    # input: a df with cols: idx, word, length, y, name, label
    df = preprocess(df)
    gt, pd = [], []
    for index, row in df.iterrows():
        gt += [row["label"], ] * row["length"]
        pd += [lexicon.index(row["word"]), ] * row["length"]
    return normalized_mutual_info_score(gt, pd)

def ngram_ent(df, n=4, lb=1):  # this is not n-gram entropy, this is a pre-processing function
    # input: a df with cols: idx, word, length, y, name
    # input is not expected to have gone through preprocessing
    # lb: filter out instance <= lb frames
    bins = {}
    dfs = {_: list(x[x["length"] > lb]["word"]) for _, x in df.groupby('y') if len(x) > 1}
    for k, v in dfs.items():
        if len(v) >= n:
            for i in range(len(v) - n + 1):
                pattern = "".join(v[i:i + n])
                if pattern in bins:
                    bins[pattern] += 1
                else:
                    bins[pattern] = 1
    return bins

def nge(df, K, n=2, lb=5):  # this is n-gram entropy
    bins = ngram_ent(df, n, lb)
    if not len(bins):
        return 0.
    assert K ** n >= len(bins)
    dist = [v for k, v in bins.items()] + [0] * (K ** n - len(bins))  # compensate for n-gram that did not appear
    ent = entropy(np.array(dist) / sum(dist), base=2)
    return ent

def metric_f2(df, K):  # this calculates the F_2 in paper, nge returns the K_n in paper
    return nge(df, K, n=2) - nge(df, K, n=1)
