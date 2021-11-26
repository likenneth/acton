import math, random
import numpy as np
import torch

def body_center(joint, dim=3, lhip_id=11, rhip_id=12):
    # TODO import lhip_id and rhip_id from dataset
    lhip = joint[lhip_id * dim:lhip_id * dim + dim]
    rhip = joint[rhip_id * dim:rhip_id * dim + dim]
    body_center = (lhip + rhip) * 0.5
    return body_center


def body_unit(joint, dim=3, lhip_id=11, rhip_id=12):
    # TODO import lhip_id and rhip_id from dataset
    lhip = joint[lhip_id * dim:lhip_id * dim + dim]
    rhip = joint[rhip_id * dim:rhip_id * dim + dim]
    unit = torch.linalg.norm(lhip - rhip, ord=2)  # positive hip width
    return unit


def euler_rodrigues_rotation(theta, axis):
    # https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    a = np.cos(theta / 2.0)
    b, c, d = - axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_matrix = torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab),
                         aa + dd - bb - cc]]).float()  # originally cosine returns double
    return rot_matrix


def time_distortion_func(a, b):
    pass


def time_scaling_func(s):
    assert s > 0


class SkeletonTransform(object):
    def __init__(self, aug_shift_prob, aug_shift_range, aug_rot_prob, aug_rot_range, min_length, max_length, aug_time_prob, aug_time_rate):
        # TODO: move to config if necessary
        self.axis = np.array([0, 0, 1])
        # axis = axis / math.sqrt(np.dot(axis, axis))
        self.norm_frame = 0
        self.dim = 3
        self.aug_shift_prob = aug_shift_prob
        self.aug_shift_range = aug_shift_range
        self.aug_rot_prob = aug_rot_prob
        self.aug_rot_range = aug_rot_range
        self.aug_time_prob = aug_time_prob
        self.aug_time_faster_prob = 0.5
        assert 1 <= aug_time_rate < 2
        self.aug_time_rate = aug_time_rate
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, x, shut=False, seed=None):
        if seed:
            random.seed(seed)
        # input and output, torch tensor of shape [T, 51]
        # returns transformed tensor, and a list of int depicting its correspondence to original index
        ttl = x.shape[0]
        joint_num = int(x.shape[-1] / self.dim)

        norm_frame = random.randint(0, ttl-1)
        # print("Norm Frame is", norm_frame)
        # norm_frame = self.norm_frame
        # spatial translation normalization by first frame
        if joint_num == 17:
            ct = body_center(x[norm_frame])
        else:
            ct = body_center(x[norm_frame], lhip_id=12, rhip_id=16)
        x -= ct.repeat(joint_num).unsqueeze(0)

        assert not x.isnan().any(), "After Translation Normalization"
        
        if joint_num == 17:
            # spatial rotation normalization by first frame
            if joint_num == 17:
                lh = x[norm_frame, 33:35]  # left hip x, left hip y
            else:
                lh = x[norm_frame, 36:38]  # left hip x, left hip y 
            theta = float(-np.arccos(lh[0] / np.sqrt(np.dot(lh, lh))))
            ttt = euler_rodrigues_rotation(theta, self.axis)
            x = (x.reshape(-1, self.dim) @ ttt.transpose(1, 0)).reshape(ttl, joint_num * self.dim)

            assert not x.isnan().any(), "After Rotation Normalization"

            # spatial rotation augmentation
            if random.random() < self.aug_rot_prob and not shut:  # let's rotate
                theta = 2 * math.pi * (random.random() - .5) * self.aug_rot_range
                # Euler-Rodrigues formula
                ttt = euler_rodrigues_rotation(theta, self.axis)
                x = (x.reshape(-1, self.dim) @ ttt.transpose(1, 0)).reshape(ttl, joint_num * self.dim)

            assert not x.isnan().any(), "After Rotation Augmentation"

        # TODO: spatial scaling augmentation

        # spatial translation augmentation
        if random.random() < self.aug_shift_prob and not shut:  # let's translate
            if joint_num == 17:
                unit = body_unit(x[self.norm_frame])
            else:
                unit = body_unit(x[self.norm_frame], lhip_id=12, rhip_id=16)
            move_x = (random.random() - .5) * self.aug_shift_range * unit
            move_y = (random.random() - .5) * self.aug_shift_range * unit
            move_z = 0  #(random.random() - .5) * self.aug_shift_range * unit  # TODO: this is very suspicious!
            shift = torch.Tensor([move_x, move_y, move_z])
            x = (x.reshape(-1, self.dim) - shift.unsqueeze(0)).reshape(ttl, joint_num * self.dim)
        
        assert not x.isnan().any(), "After Translation Augmentation"

        # temporal distortion augmentation
        # The index of velo is the index of distorted video while the value of velo is the index of original video.
        # It's an index to index pair
        velo = np.arange(ttl)
        # requirement 1: len(velo) == x.size(0)
        # requirement 2: velo[0] == 0 and velo[-1] == ttl - 1
        if random.random() < self.aug_time_prob and not shut:  # let's do velocity augmentation
            # uniform (1, self.aug_time_rate)
            t_scale_ = (1 - self.aug_time_rate) * random.random() + self.aug_time_rate
            if random.random() < 0.5:  # slower
                t_scale = t_scale_
            else:  # faster
                t_scale = 1.0 / t_scale_

            # slower or faster
            # t_scale = 2 * (self.aug_time_rate - 1) * random.random() + 2 - self.aug_time_rate
            # only slower
            # t_scale = (self.aug_time_rate - 1) * random.random() + 1
            # only faster
            # t_scale = (self.aug_time_rate - 1) * random.random() + 2 - self.aug_time_rate

            new_ttl = int(ttl * t_scale + 0.5)
            new_velo = np.arange(new_ttl)
            # TODO: this is not exact nearest neighbor. +0.5 may cause new_velo[-1] == ttl. deal with it later
            new_velo = new_velo / t_scale
            # assert new_velo[-1] <= ttl - 1
            new_x = x[np.floor(new_velo).astype(int)]

            # if self.aug_time_rate == 1:
            #     assert ttl == new_ttl
            #     assert (velo - new_velo).sum().item() < 1e-12
            #     assert (x - new_x).sum().item() < 1e-12

            ttl = new_ttl
            velo = new_velo
            x = new_x
        assert not x.isnan().any(), "After Temporal Augmentation"

        return x, velo
