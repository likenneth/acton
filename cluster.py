import argparse
import os
import pprint
import shutil
import time
import sys
import yaml
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from src.data.dataset.loader import AISTDataset
from src import algo
from src.data.dataset.cluster_misc import lexicon, get_names, genre_list

from plb.models.self_supervised import TAN
from plb.models.self_supervised.tan import TANEvalDataTransform, TANTrainDataTransform
from plb.datamodules import SeqDataModule
from plb.datamodules.data_transform import body_center, euler_rodrigues_rotation

KEYPOINT_NAME = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                 "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                 "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

import pytorch_lightning as pl
pl.utilities.seed.seed_everything(0)

def plain_distance(a, b):
    return np.linalg.norm(a - b, ord=2)

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--data_dir',
                        help='path to aistplusplus data directory from repo root',
                        type=str)
    
    parser.add_argument('--seed',
                        help='seed for this run',
                        default=1,
                        type=int)

    args, _ = parser.parse_known_args()
    pl.utilities.seed.seed_everything(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.cfg, 'r') as stream:
        ldd = yaml.safe_load(stream)

    if args.data_dir:
        ldd["PRETRAIN"]["DATA"]["DATA_DIR"] = args.data_dir
    pprint.pprint(ldd)
    return ldd

def main():
    args = parse_args()
    debug = args["NAME"] == "debug"
    log_dir = os.path.join("./logs", args["NAME"])

    dirpath = Path(log_dir)
    dirpath.mkdir(parents=True, exist_ok=True)

    timed = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(log_dir, f"config_used_{timed}.yaml"), "w") as stream:
        yaml.dump(args, stream, default_flow_style=False)
    video_dir = os.path.join(log_dir, "saved_videos")
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    official_loader = AISTDataset(os.path.join(args["PRETRAIN"]["DATA"]["DATA_DIR"], "annotations"))

    if 1:  # change this to 0 for skeleton experiment
        # get model
        load_name = args["CLUSTER"]["CKPT"] if args["CLUSTER"]["CKPT"] != -1 else args["NAME"]
        with open(os.path.join(log_dir, f"val_cluster_zrsc_scores.txt"), "a") as f:
            f.write(f"EXP: {load_name}\n")
        cfg = None
        for fn in os.listdir(os.path.join("./logs", load_name)):
            if fn.endswith(".yaml"):
                cfg = fn
        with open(os.path.join("./logs", load_name, cfg), 'r') as stream:
            old_args = yaml.safe_load(stream)
        cpt_name = os.listdir(os.path.join("./logs", load_name, "default/version_0/checkpoints"))[0]
        print(f"We are using checkpoint: {cpt_name}")
        model = eval(old_args["PRETRAIN"]["ALGO"]).load_from_checkpoint(os.path.join("./logs", load_name, "default/version_0/checkpoints", cpt_name))
        model.eval()
        def ske2feat(ldd):
            ldd1 = torch.Tensor(ldd).flatten(1, -1) / 100  # [T, 51]
            ttl = ldd1.shape[0]
            ct = body_center(ldd1[0])
            ldd1 -= ct.repeat(17).unsqueeze(0)
            res1 = model(ldd1.unsqueeze(0), torch.tensor([ttl]))
            forward_feat = res1[:, 0]  # [T1, f]
            forward_feat /= torch.linalg.norm(forward_feat, dim=-1, keepdim=True, ord=2)
            return forward_feat
    else:
        # to get results for using raw skeleton, swap with
        def ske2feat(ldd):
            ldd1 = torch.Tensor(ldd).flatten(1, -1) / 100  # [T, 51]
            ttl = ldd1.shape[0]
            ct = body_center(ldd1[0])
            ldd1 -= ct.repeat(17).unsqueeze(0)
            return ldd1

    # get data
    tr_kpt_container = []
    tr_len_container = []
    tr_feat_container = []
    tr_name_container = []
    val_kpt_container = []
    val_len_container = []
    val_feat_container = []
    val_name_container = []
    for genre in genre_list:  # mix every genre together
        # train data, we only have training set in this setting
        tr_df = get_names(genre, trval="train", seed=4321)
        tr_df = tr_df[tr_df["situ"] == "sFM"]
        val_df = get_names(genre, trval="val", seed=4321)
        val_df = val_df[val_df["situ"] == "sFM"]
        for reference_name in tqdm(list(tr_df["name"]), desc='Loading training set features'):
            ldd = official_loader.load_keypoint3d(reference_name)
            tr_kpt_container.append(ldd)
            tr_len_container.append(ldd.shape[0])
            tr_feat_container.append(ske2feat(ldd).detach().cpu().numpy())
            tr_name_container.append(reference_name)
        for reference_name in tqdm(list(val_df["name"]), desc='Loading validation set features'):
            ldd = official_loader.load_keypoint3d(reference_name)
            val_kpt_container.append(ldd)
            val_len_container.append(ldd.shape[0])
            val_feat_container.append(ske2feat(ldd).detach().cpu().numpy())
            val_name_container.append(reference_name)

    tr_where_to_cut = [0, ] + list(np.cumsum(np.array(tr_len_container)))
    tr_stacked = np.vstack(tr_feat_container)
    val_where_to_cut = [0, ] + list(np.cumsum(np.array(val_len_container)))
    val_stacked = np.vstack(val_feat_container)

    for K in range(args["CLUSTER"]["K_MIN"], args["CLUSTER"]["K_MAX"], 10):
        argument_dict = {"distance": plain_distance, "TYPE": "vanilla", "K": K, "TOL": 1e-4}
        if not os.path.exists(os.path.join(log_dir, f"advanced_centers_{K}.npy")):
            c = getattr(algo, args["CLUSTER"]["TYPE"])(tr_stacked, times=args["CLUSTER"]["TIMES"], argument_dict=argument_dict)
            np.save(os.path.join(log_dir, f"advanced_centers_{K}.npy"), c.kmeans.cluster_centers_)
        else:
            ctrs = np.load(os.path.join(log_dir, f"advanced_centers_{K}.npy"))
            c = getattr(algo, args["CLUSTER"]["TYPE"] + "_clusterer")(TIMES=args["CLUSTER"]["TIMES"], K=K, TOL=1e-4)
            c.fit(tr_stacked[:K])
            c.kmeans.cluster_centers_ = ctrs
        # infer on training set and save
        y = np.concatenate([np.ones((l,)) * i for i, l in enumerate(tr_len_container)], axis=0)
        s = np.concatenate([np.arange(l) for i, l in enumerate(tr_len_container)], axis=0)
        tr_res_df = pd.DataFrame(y, columns=["y"])  # from which sequence
        cluster_l = c.get_assignment(tr_stacked)  # assigned to which cluster
        tr_res_df['cluster'] = cluster_l
        tr_res_df['frame_index'] = s  # the frame index in home sequence
        tr_word_df = pd.DataFrame(columns=["idx", "word", "length", "y", "name"])  # word index in home sequence
        for sequence_idx in range(len(tr_len_container)):
            name = tr_name_container[sequence_idx]
            cluster_seq = list(cluster_l[tr_where_to_cut[sequence_idx]: tr_where_to_cut[sequence_idx + 1]]) + [-1, ]
            running_idx = 0
            prev = -1
            current_len = 0
            for cc in cluster_seq:
                if cc == prev:
                    current_len += 1
                else:
                    tr_word_df = tr_word_df.append(
                        {"idx": int(running_idx), "word": lexicon[prev], "length": current_len, "y": sequence_idx,
                         "name": name}, ignore_index=True)
                    running_idx += 1
                    current_len = 1
                prev = cc
        tr_word_df = tr_word_df[tr_word_df["idx"] > 0]
        tr_word_df.to_pickle(dirpath / f"advanced_tr_{K}.pkl")
        print(f"advanced_tr_{K}.pkl dumped to {log_dir}")  # saved tokenization of training set

        # infer on validation set and save
        y = np.concatenate([np.ones((l,)) * i for i, l in enumerate(val_len_container)], axis=0)
        s = np.concatenate([np.arange(l) for i, l in enumerate(val_len_container)], axis=0)
        val_res_df = pd.DataFrame(y, columns=["y"])  # from which sequence
        cluster_l = c.get_assignment(val_stacked)  # assigned to which cluster
        val_res_df['cluster'] = cluster_l
        val_res_df['frame_index'] = s  # the frame index in home sequence
        val_word_df = pd.DataFrame(columns=["idx", "word", "length", "y", "name"])  # word index in home sequence
        for sequence_idx in range(len(val_len_container)):
            name = val_name_container[sequence_idx]
            cluster_seq = list(cluster_l[val_where_to_cut[sequence_idx]: val_where_to_cut[sequence_idx + 1]]) + [-1, ]
            running_idx = 0
            prev = -1
            current_len = 0
            for cc in cluster_seq:
                if cc == prev:
                    current_len += 1
                else:
                    val_word_df = val_word_df.append(
                        {"idx": int(running_idx), "word": lexicon[prev], "length": current_len, "y": sequence_idx,
                         "name": name}, ignore_index=True)
                    running_idx += 1
                    current_len = 1
                prev = cc
        val_word_df = val_word_df[val_word_df["idx"] > 0]
        val_word_df.to_pickle(dirpath / f"advanced_val_{K}.pkl")
        print(f"advanced_val_{K}.pkl dumped to {log_dir}")  # saved tokenization of validation set

if __name__ == '__main__':
    main()
