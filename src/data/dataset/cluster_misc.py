import os
import json
import string
import itertools
import pandas as pd
from tqdm import tqdm

all26 = list(string.ascii_lowercase)
lexicon = []
for (c, v, s) in itertools.permutations(all26, 3):
    lexicon.append(c + v + s)


keys = ["genre", "situ", "dancer", "tempo", "choreo", "name"]
genre_list = ["gBR", "gPO", "gLO", "gMH", "gLH", "gHO", "gWA", "gKR", "gJS", "gJB"]

def vidn_parse(s):
    res = {}
    if s.endswith("pkl"):
        s = s[:-4]
    for seg in s.split("_"):
        if seg.startswith("g"):
            res["genre"] = seg
        elif seg.startswith("s"):
            res["situ"] = seg
        elif seg.startswith("d"):
            res["dancer"] = seg
        elif seg.startswith("m"):
            res["tempo"] = 10 * (int(seg[3:]) + 8)
        elif seg.startswith("ch"):
            res["choreo"] = res["genre"][1:] + seg[2:] + res["situ"][1:]
        else:
            pass
    res["name"] = s  # currently does not support camera variation
    return res

def get_names(genre, trval="train", seed=1234):
    validation_split = []
    with open(os.path.join("src/data/dataset", "splits", f"split_wseed_{seed}.json"), "r") as f:
        ldd = json.load(f)
        validation_split += ldd[genre]

    annot_3d = list(os.listdir("../aistplusplus/annotations/keypoints3d"))
    filter_file = os.path.join("../aistplusplus/annotations/", 'ignore_list.txt')
    with open(filter_file, "r") as f:
        filter_file = [_[:-1] for _ in f.readlines()]
    annot_3d = [_ for _ in annot_3d if _.startswith(genre) and _[:-4] not in filter_file]

    res = pd.DataFrame(columns=keys)
    if trval == "train" or trval == "tr":
        for s in annot_3d:
            if not s[:-4] in validation_split:  # only keep samples in the training set
                res = res.append(vidn_parse(s), ignore_index=True)
    elif trval == "val":
        for s in annot_3d:
            if s[:-4] in validation_split:  # only keep samples in the validation set
                res = res.append(vidn_parse(s), ignore_index=True)
    else:
        raise NotImplementedError

    return res.sort_values(["situ", "choreo"])

def get_num_in_table(df, names, signs):
    # a df filtered to have only one exp
    dfs = {_: x for _, x in df.groupby("genre") if len(x) > 1}
    containers = [[] for name in names]

    for genre, little_df in dfs.items():
        for i, name in enumerate(names):
            nmi_list = list(little_df[little_df["type"]==name]["value"])
            if len(nmi_list):
                containers[i].append(sum(nmi_list) / len(nmi_list))
            else:
                containers[i].append(-9999)

    res = []
    for container, sign in zip(containers, signs):
        nmi = sum(container) / len(container) * sign
        res.append(nmi)
    return res

if __name__ == "__main__":
    print(len(lexicon))
    print(lexicon)