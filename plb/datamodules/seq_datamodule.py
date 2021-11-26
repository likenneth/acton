import multiprocessing
import random
import os
import numpy as np
from typing import Any, Callable, Optional
from tqdm import tqdm

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .dataset import SkeletonDataset
from .data_transform import SkeletonTransform

num_cpu = multiprocessing.cpu_count()
preciser = True

# mp worker
def get_from_name(loader, seq_name):
    return seq_name, loader.load_keypoint3d(seq_name)


def select_valid_3D_skeletons(skeletons, thre):
    # fresh out of official loader is [T, 17, 3] in centinetre
    max_per_frame = np.max(np.max(skeletons, axis=-1), axis=-1)  # [T]
    sel = max_per_frame < thre
    kicks = np.sum(~(sel)).item()
    return sel, kicks

miu = 3
scale = 1
def gaussian(m):
    # input a torch tensor with the un-abs-ed distance
    # returns a Gaussian value of the same shape
    m = torch.abs(m)
    sh = - torch.square(m / miu) / 2
    ex = torch.exp(sh) * scale
    return ex

def two_trans_collate(lop):
    first_len = torch.tensor([len(_[2]) for _ in lop])
    second_len = torch.tensor([len(_[3]) for _ in lop])
    first_max = torch.max(first_len).item()
    second_max = torch.max(second_len).item()

    first_container = []
    second_container = []
    # first_container_velo = []
    # second_container_velo = []
    rect_container = []
    view1_container = []
    view2_container = []
    indices1 = []
    indices2 = []
    chopped_bs = []
    for b, (i1, i2, velo1, velo2) in enumerate(lop):
        first_container.append(torch.cat([i1, torch.zeros(size=(first_max - len(velo1), i1.shape[-1]))], dim=0))
        second_container.append(torch.cat([i2, torch.zeros(size=(second_max - len(velo2), i2.shape[-1]))], dim=0))
        # first_container_velo.append(torch.cat([torch.tensor(velo1), torch.zeros(size=(first_max - len(velo1),))], dim=0))
        # second_container_velo.append(torch.cat([torch.tensor(velo2), torch.zeros(size=(second_max - len(velo2),))], dim=0))
        # creating a rectangle for loss calculation, pending a lot of heuristic design
        dist = torch.tensor(velo1).unsqueeze(-1) - torch.tensor(velo2).unsqueeze(0)  # [t1, t2]
        rect = gaussian(dist).float()
        rect_container.append(rect)

        dist = torch.tensor(velo1).unsqueeze(-1) - torch.tensor(velo1).unsqueeze(0)  # [t1, t1]
        rect = gaussian(dist).float()
        view1_container.append(rect)

        dist = torch.tensor(velo2).unsqueeze(-1) - torch.tensor(velo2).unsqueeze(0)  # [t2, t2]
        rect = gaussian(dist).float()
        view2_container.append(rect)

        floor_velo_dist = torch.tensor(np.floor(velo1).astype(int)).unsqueeze(-1) - torch.tensor(np.floor(velo2).astype(int)).unsqueeze(0)
        loc1, loc2 = torch.where(floor_velo_dist == 1)
        if preciser:
            # here we remove even more positives
            keep = []
            prev = -1
            luck = []
            for i in range(loc1.shape[0]):
                if loc1[i] == prev:
                    luck.append(i)
                else:
                    if len(luck):
                        chosen = random.choice(luck)
                        keep.append(chosen)
                    luck = [i]
                prev = loc1[i]
            chosen = random.choice(luck)
            keep.append(chosen)
            loc1, loc2 = loc1[keep], loc2[keep]
            keep = []
            prev = -1
            luck = []
            for i in range(loc2.shape[0]):
                if loc2[i] == prev:
                    luck.append(i)
                else:
                    if len(luck):
                        chosen = random.choice(luck)
                        keep.append(chosen)
                    luck = [i]
                prev = loc1[i]
            chosen = random.choice(luck)
            keep.append(chosen)
            loc1, loc2 = loc1[keep], loc2[keep]

        indices1.append(loc1 + b * first_max)
        indices2.append(loc2 + b * second_max)
        chopped_bs.append(loc1.size(0))
    indices1 = torch.cat(indices1, dim=0)
    indices2 = torch.cat(indices2, dim=0)
    assert len(indices1) == len(indices2)

    first_view = torch.stack(first_container)
    second_view = torch.stack(second_container)
    # first_velo = torch.stack(first_container_velo)
    # second_velo = torch.stack(second_container_velo)
    m = torch.block_diag(*rect_container)
    v1 = torch.block_diag(*view1_container)
    v2 = torch.block_diag(*view2_container)
    # m = torch.cat([torch.cat([v1, m], dim=1), torch.cat([m.t(), v2], dim=1)], dim=0)
    m = torch.cat([torch.cat([v1.new_zeros(v1.shape), m], dim=1), torch.cat([m.t(), v2.new_zeros(v2.shape)], dim=1)], dim=0)
    chopped_bs = torch.tensor(chopped_bs)

    return first_view, second_view, first_len, second_len, m, indices1, indices2, chopped_bs


class SeqDataset(Dataset):
    def __init__(self, data, transform):
        # data: a list of torch tensor, each of shape [T, 51]
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # innately has some randomness, will crop a continuous chunk from a video of batch size length
        tbc = self.data[item]
        return self.transform(tbc)


class SeqDataModule(LightningDataModule):
    name = 'seq'

    def __init__(
            self,
            DATA_DIR,
            GENRE,
            SPLIT,
            BS,
            AUG_SHIFT_PROB,
            AUG_SHIFT_RANGE,
            AUG_ROT_PROB,
            AUG_ROT_RANGE,
            MIN_LENGTH,
            MAX_LENGTH,
            NUM_WORKERS,
            AUG_TIME_PROB,
            AUG_TIME_RATE,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = SkeletonDataset(DATA_DIR, GENRE, SPLIT)
        self.batch_size = BS  # the batch size to show to outer modules
        self.aug_shift_prob = AUG_SHIFT_PROB
        self.aug_shift_range = AUG_SHIFT_RANGE
        self.aug_rot_prob = AUG_ROT_PROB
        self.aug_rot_range = AUG_ROT_RANGE
        self.min_length = MIN_LENGTH
        self.max_length = MAX_LENGTH
        self.train_data = []
        self.val_data = []
        self.num_samples = 0  # not the real num samples, but the total frame number in training set
        self.num_samples_valid = 0
        self.num_workers = NUM_WORKERS
        self.aug_time_prob = AUG_TIME_PROB
        self.aug_time_rate = AUG_TIME_RATE
        self.prepare_data()

        # TODO: move data washing into self.dataset?
        # TODO: deal with constant
        self.num_proc = self.num_workers if self.num_workers > 0 else 1
        self.threshold = 500
        name_list = self.dataset.train_split + self.dataset.validation_split
        with multiprocessing.Pool(self.num_proc) as p:
            for name, res in p.starmap(get_from_name,
                                       tqdm(zip([self.dataset.official_loader] * len(name_list), name_list),
                                            total=len(name_list), desc='Loading training data...', leave=True)):
                sel, kicks = select_valid_3D_skeletons(res, self.threshold)
                if kicks > 0:
                    print(f"kicking out {kicks} frames out of {name} for train, threshold is {self.threshold}")
                    res = res[sel]
                # let's normalize for numerical stability
                res = res / 100.0  # originally in cm, now in m

                if SPLIT == 4321:
                    if name in self.dataset.train_split and "sFM" in name:
                        self.train_data.append(torch.tensor(res).float().flatten(1))
                        self.num_samples += 1  # related to simCLR or scheduling, important
                    elif name in self.dataset.validation_split and "sFM" in name:
                        self.val_data.append(torch.tensor(res).float().flatten(1))
                        self.num_samples_valid += 1
                    else:
                        pass
                elif SPLIT == 1234:
                    if "sFM" in name:
                        self.train_data.append(torch.tensor(res).float().flatten(1))
                        self.num_samples += 1  # related to simCLR or scheduling, important
                    elif name in self.dataset.validation_split:
                        self.val_data.append(torch.tensor(res).float().flatten(1))
                        self.num_samples_valid += 1
                    else:
                        pass
                else:
                    assert 0, 'unknown split, should be in [1234, 4321]'

        print(
            f"SPLIT {SPLIT} dances loaded with {self.num_samples} training videos and {self.num_samples_valid} validation videos")

    def train_dataloader(self) -> DataLoader:
        train_dataset = SeqDataset(self.train_data, transform=self.train_transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=two_trans_collate,
                                      num_workers=self.num_workers)
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataset = SeqDataset(self.val_data, transform=self.val_transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, collate_fn=two_trans_collate,
                                    num_workers=self.num_workers)
        return val_dataloader

    def _default_transforms(self) -> Callable:
        data_transforms = SkeletonTransform(self.aug_shift_prob, self.aug_shift_range, self.aug_rot_prob, self.aug_rot_range, self.min_length, self.max_length, self.aug_time_prob, self.aug_time_rate)
        return data_transforms
