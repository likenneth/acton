# coding=utf-8
# Copyright 2020 The Google AI Perception Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Visualize the AIST++ Dataset."""

from . import utils
import cv2
import numpy as np

_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
           [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
           [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
           [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
           [255, 0, 170], [255, 0, 85]]

_COLORS_BONE = [
    ([0, 1], [255, 0, 0]), ([0, 2], [255, 0, 0]),
    ([1, 3], [255, 170, 0]), ([2, 4], [255, 170, 0]),
    ([5, 6], [0, 255, 255]), ([11, 12], [0, 255, 255]),  # the two bridge
    ([6, 8], [0, 255, 255]), ([8, 10], [0, 255, 255]),
    ([6, 12], [0, 255, 255]), ([12, 14], [0, 255, 255]), ([14, 16], [0, 255, 255]),
    ([5, 7], [255, 0, 85]), ([7, 9], [255, 0, 85]),
    ([5, 11], [255, 0, 85]), ([11, 13], [255, 0, 85]), ([13, 15], [255, 0, 85]),
]

_COLORS_BONE_PLAIN = [
    ([0, 1], [0, 0, 0]), ([0, 2], [0, 0, 0]),
    ([1, 3], [0, 0, 0]), ([2, 4], [0, 0, 0]),
    ([5, 6], [0, 0, 0]), ([11, 12], [0, 0, 0]),  # the two bridge
    ([6, 8], [0, 0, 0]), ([8, 10], [0, 0, 0]),
    ([6, 12], [0, 0, 0]), ([12, 14], [0, 0, 0]), ([14, 16], [0, 0, 0]),
    ([5, 7], [0, 0, 0]), ([7, 9], [0, 0, 0]),
    ([5, 11], [0, 0, 0]), ([11, 13], [0, 0, 0]), ([13, 15], [0, 0, 0]),
]


def plot_kpt(keypoint, canvas, bones=True):
    if bones:
        for j, c in _COLORS_BONE:
            cv2.line(canvas,
                     tuple(keypoint[:, 0:2][j[0]].astype(int).tolist()),
                     tuple(keypoint[:, 0:2][j[1]].astype(int).tolist()),
                     tuple(c), thickness=2)
    else:
        for i, (x, y) in enumerate(keypoint[:, 0:2]):
            if np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(canvas, (int(x), int(y)), 7, _COLORS[i % len(_COLORS)], thickness=-1)
    return canvas

def plot_kpt_plain(keypoint, canvas, c, bones=True):
    if bones:
        for j, _ in _COLORS_BONE_PLAIN:
            cv2.line(canvas,
                     tuple(keypoint[:, 0:2][j[0]].astype(int).tolist()),
                     tuple(keypoint[:, 0:2][j[1]].astype(int).tolist()),
                     tuple(c), thickness=2)
    else:
        for i, (x, y) in enumerate(keypoint[:, 0:2]):
            if np.isnan(x) or np.isnan(y):
                continue
            cv2.circle(canvas, (int(x), int(y)), 7, _COLORS[i % len(_COLORS)], thickness=-1)
    return canvas


def plot_on_video(keypoints2d, video_path, save_path, fps=60):
    assert len(keypoints2d.shape) == 3, (
        f'Input shape is not valid! Got {keypoints2d.shape}')
    video = utils.ffmpeg_video_read(video_path, fps=fps)
    for iframe, keypoint in enumerate(keypoints2d):
        if iframe >= video.shape[0]:
            break
        video[iframe] = plot_kpt(keypoint, video[iframe])
    utils.ffmpeg_video_write(video, save_path, fps=fps)
