import argparse
import os
import pprint
import sys

import json
import yaml
import shutil
import time
import logging, json
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from plb.models.self_supervised import TAN
from plb.models.self_supervised.tan import TANEvalDataTransform, TANTrainDataTransform
from plb.datamodules import SeqDataModule
from pytorch_lightning.plugins import DDPPlugin

KEYPOINT_NAME = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                 "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                 "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

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

    # log
    tt_logger = TestTubeLogger(
        save_dir=log_dir,
        name="default",
        debug=False,
        create_git_tag=False
    )

    # trainer
    trainer = pl.Trainer(
        gpus=args["PRETRAIN"]["GPUS"],
        check_val_every_n_epoch=args["PRETRAIN"]["TRAINER"]["VAL_STEP"],
        logger=tt_logger,
        accelerator=args["PRETRAIN"]["TRAINER"]["ACCELERATOR"],
        max_epochs=args["PRETRAIN"]["EPOCH"],
        gradient_clip_val=0.5,
        num_sanity_val_steps=0,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    j = 17
    dm = SeqDataModule(**args["PRETRAIN"]["DATA"])
    transform_args = {"min_length": args["PRETRAIN"]["DATA"]["MIN_LENGTH"],
                      "max_length": args["PRETRAIN"]["DATA"]["MAX_LENGTH"],
                      "aug_shift_prob": args["PRETRAIN"]["DATA"]["AUG_SHIFT_PROB"],
                      "aug_shift_range": args["PRETRAIN"]["DATA"]["AUG_SHIFT_RANGE"],
                      "aug_rot_prob": args["PRETRAIN"]["DATA"]["AUG_ROT_PROB"],
                      "aug_rot_range": args["PRETRAIN"]["DATA"]["AUG_ROT_RANGE"],
                      "aug_time_prob": args["PRETRAIN"]["DATA"]["AUG_TIME_PROB"],
                      "aug_time_rate": args["PRETRAIN"]["DATA"]["AUG_TIME_RATE"], }
    dm.train_transforms = eval(args["PRETRAIN"]["ALGO"] + "TrainDataTransform")(**transform_args)
    dm.val_transforms = eval(args["PRETRAIN"]["ALGO"] + "EvalDataTransform")(**transform_args)
    model = eval(args["PRETRAIN"]["ALGO"])(
        gpus=args["PRETRAIN"]["GPUS"],
        num_samples=dm.num_samples,
        batch_size=dm.batch_size,
        length=dm.min_length,
        dataset=dm.name,
        max_epochs=args["PRETRAIN"]["EPOCH"],
        warmup_epochs=args["PRETRAIN"]["WARMUP"],
        arch=args["PRETRAIN"]["ARCH"]["ARCH"],
        val_configs=args["PRETRAIN"]["VALIDATION"],
        learning_rate=float(args["PRETRAIN"]["TRAINER"]["LR"]),
        log_dir=log_dir,
        protection=args["PRETRAIN"]["PROTECTION"],
        optim=args["PRETRAIN"]["TRAINER"]["OPTIM"],
        lars_wrapper=args["PRETRAIN"]["TRAINER"]["LARS"],
        tr_layer=args["PRETRAIN"]["ARCH"]["LAYER"],
        tr_dim=args["PRETRAIN"]["ARCH"]["DIM"],
        neg_dp=args["PRETRAIN"]["ARCH"]["DROPOUT"],
        j=j*3, 
    )

    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    main()
