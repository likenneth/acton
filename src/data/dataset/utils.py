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
"""Utils for AIST++ Dataset."""
import os
import json
import ffmpeg
import numpy as np
import contextlib
from PIL import Image, ImageDraw, ImageFont

import aniposelib

from src.data.dataset.visualizer import plot_kpt, plot_kpt_plain
from src.data.dataset.cluster_misc import lexicon, get_names, genre_list, vidn_parse

font = ImageFont.truetype("alata.ttf", 12)
font_large = ImageFont.truetype("alata.ttf", 24)
CAP_COL = (16, 16, 16)

parse_keys = ["genre", "situ", "dancer", "tempo", "choreo", "name"]
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

def ffmpeg_video_read(video_path, fps=None):
    """Video reader based on FFMPEG.

    This function supports setting fps for video reading. It is critical
    as AIST++ Dataset are constructed under exact 60 fps, while some of
    the AIST dance videos are not percisely 60 fps.

    Args:
      video_path: A video file.
      fps: Use specific fps for video reading. (optional)
    Returns:
      A `np.array` with the shape of [seq_len, height, width, 3]
    """
    assert os.path.exists(video_path), f'{video_path} does not exist!'
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    video_info = next(stream for stream in probe['streams']
                      if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    stream = ffmpeg.input(video_path)
    if fps:
        stream = ffmpeg.filter(stream, 'fps', fps=fps, round='up')
    stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
    out, _ = ffmpeg.run(stream, capture_stdout=True)
    out = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return out.copy()


def ffmpeg_video_write(data, video_path, fps=25):
    """Video writer based on FFMPEG.

    Args:
      data: A `np.array` with the shape of [seq_len, height, width, 3]
      video_path: A video file.
      fps: Use specific fps for video writing. (optional)
    """
    assert len(data.shape) == 4, f'input shape is not valid! Got {data.shape}!'
    _, height, width, _ = data.shape
    # import pdb; pdb.set_trace()
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = (
        ffmpeg
            .input('pipe:', framerate=fps, format='rawvideo',
                   pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(video_path, pix_fmt='yuv420p')
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in data:
        writer.stdin.write(frame.astype(np.uint8).tobytes())
    writer.stdin.close()


def save_keypoints3d_as_video(keypoints3d, captions, data_root, video_path):
    assert len(captions) == keypoints3d.shape[0]
    file_path = os.path.join(data_root, "annotations/cameras/setting1.json")
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'r') as f:
        params = json.load(f)
    cameras = []
    for param_dict in params:
        camera = aniposelib.cameras.Camera(name=param_dict['name'],
                                           size=param_dict['size'],
                                           matrix=param_dict['matrix'],
                                           rvec=param_dict['rotation'],
                                           tvec=param_dict['translation'],
                                           dist=param_dict['distortions'])
        cameras.append(camera)
    cgroup = aniposelib.cameras.CameraGroup(cameras)
    length = keypoints3d.shape[0]
    keypoints2d = (cgroup.project(keypoints3d) // 2).reshape(9, length, 17, 2)[0]
    blank_video = np.ones((length, 1080 // 2, 1920 // 2, 3), dtype=np.uint8) * 255
    for iframe, (keypoint, cap) in enumerate(zip(keypoints2d, captions)):
        if iframe >= blank_video.shape[0]:
            break
        tmp = plot_kpt(keypoint, blank_video[iframe])
        tmp = Image.fromarray(tmp, 'RGB')
        ImageDraw.Draw(tmp).text((25, 25), cap, CAP_COL, font=font)
        blank_video[iframe] = np.array(tmp)
    ffmpeg_video_write(blank_video, video_path, fps=30)  # play it slowly

def plot_cool(kpt, cap, data_root, c=None):
    # kpt: T, 17, 3
    file_path = os.path.join(data_root, "annotations/cameras/setting1.json")
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'r') as f:
        params = json.load(f)
    cameras = []
    for param_dict in params:
        camera = aniposelib.cameras.Camera(name=param_dict['name'],
                                           size=param_dict['size'],
                                           matrix=param_dict['matrix'],
                                           rvec=param_dict['rotation'],
                                           tvec=param_dict['translation'],
                                           dist=param_dict['distortions'])
        cameras.append(camera)
    cgroup = aniposelib.cameras.CameraGroup(cameras)
    length = kpt.shape[0]
    keypoints2d = (cgroup.project(kpt) // 2).reshape(9, length, 17, 2)[0]
    blank_video = np.ones((1080 // 2, 1920 // 2, 3), dtype=np.uint8) * 255
    for iframe, keypoint in enumerate(keypoints2d):
        if c is not None:
            blank_video = plot_kpt_plain(keypoint, blank_video, c)
        else:
            blank_video = plot_kpt_plain(keypoint, blank_video, [0, 0, 0])
        blank_video  = (255 - (255 - blank_video) * 0.8).astype(np.uint8)
    tmp = Image.fromarray(blank_video, 'RGB')
    ImageDraw.Draw(tmp).text((25, 25), cap, CAP_COL, font=font)
    return np.array(tmp)

def save_paired_keypoints3d_as_video(keypoints3d_raw, keypoints3d_gen, cap1, cap2, data_root, video_path, align=False):
    # align option will align the generated video to
    assert len(cap1) == keypoints3d_raw.shape[0] == len(cap2) == keypoints3d_gen.shape[0]

    file_path = os.path.join(data_root, "annotations/cameras/setting1.json")
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'r') as f:
        params = json.load(f)
    cameras = []
    for param_dict in params:
        camera = aniposelib.cameras.Camera(name=param_dict['name'],
                                           size=param_dict['size'],
                                           matrix=param_dict['matrix'],
                                           rvec=param_dict['rotation'],
                                           tvec=param_dict['translation'],
                                           dist=param_dict['distortions'])
        cameras.append(camera)
    cgroup = aniposelib.cameras.CameraGroup(cameras)
    length = keypoints3d_raw.shape[0]
    keypoints2d_raw = cgroup.project(keypoints3d_raw).reshape(9, length, 17, 2)[0] // 2  # there are nine cameras there
    keypoints2d_gen = cgroup.project(keypoints3d_gen).reshape(9, length, 17, 2)[0] // 2

    blank_video_raw = np.ones((length, 1080 // 2, 1920 // 2, 3), dtype=np.uint8) * 255
    for iframe, (keypoint, cap) in enumerate(zip(keypoints2d_raw, cap1)):
        if iframe >= blank_video_raw.shape[0]:
            break
        tmp = plot_kpt(keypoint, blank_video_raw[iframe])
        # import pdb; pdb.set_trace()
        tmp = Image.fromarray(tmp, 'RGB')
        ImageDraw.Draw(tmp).text((25, 25), "Raw video: "+cap, CAP_COL, font=font)
        blank_video_raw[iframe] = np.array(tmp)

    blank_video_gen = np.ones((length, 1080 // 2, 1920 // 2, 3), dtype=np.uint8) * 255
    for iframe, (keypoint, cap) in enumerate(zip(keypoints2d_gen, cap2)):
        if iframe >= blank_video_gen.shape[0]:
            break
        if cap == "no matched":
            tmp = np.ones((1080 // 2, 1920 // 2, 3), dtype=np.uint8) * 0
            tmp = Image.fromarray(tmp, 'RGB')
            ImageDraw.Draw(tmp).text((25, 25), "Gnd video: "+cap, CAP_COL, font=font)
            blank_video_gen[iframe] = np.array(tmp)
        else:
            tmp = plot_kpt(keypoint, blank_video_gen[iframe])
            tmp = Image.fromarray(tmp, 'RGB')
            ImageDraw.Draw(tmp).text((25, 25), "Gnd video: "+cap, CAP_COL, font=font)
            blank_video_gen[iframe] = np.array(tmp)

    blank_video = np.concatenate([blank_video_raw, blank_video_gen], axis=1)
    ffmpeg_video_write(blank_video, video_path, fps=30)


def save_centroids_as_video(skes, caps, data_root, video_path):
    # skes: dict of word: list of skeleton motions [T, 17 * 3]
    # caps: dict of word: list of strings
    file_path = os.path.join(data_root, "annotations/cameras/setting1.json")
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'r') as f:
        params = json.load(f)
    cameras = []
    for param_dict in params:
        camera = aniposelib.cameras.Camera(name=param_dict['name'],
                                           size=param_dict['size'],
                                           matrix=param_dict['matrix'],
                                           rvec=param_dict['rotation'],
                                           tvec=param_dict['translation'],
                                           dist=param_dict['distortions'])
        cameras.append(camera)
    cgroup = aniposelib.cameras.CameraGroup(cameras)

    blank_video_container = []
    for word in lexicon:
        if word in skes:
            for chunk_idx in range(len(skes[word])//9):  # TODO: here we sometime discard some
                near_points = []
                len_contatiner = []
                for _ in skes[word][chunk_idx*9:(chunk_idx+1)*9]:
                    length = _.shape[0]
                    len_contatiner.append(length)
                    near_points.append(cgroup.project(_).reshape(9, length, 17, 2)[0] // 2)
                length = max(len_contatiner)
                blank_video_raw = np.ones((length, 1080 // 2, 1920 // 2, 3), dtype=np.uint8) * 255
                plotted = []
                for idx, point in enumerate(near_points):
                    blank_video_gen = np.ones((length, 1080 // 2, 1920 // 2, 3), dtype=np.uint8) * 255
                    for iframe, keypoint in enumerate(point):
                        if iframe >= blank_video_gen.shape[0]:
                            break
                        tmp = plot_kpt(keypoint, blank_video_gen[iframe])
                        tmp = Image.fromarray(tmp, 'RGB')
                        ImageDraw.Draw(tmp).text((25, 25), f"Near-by ground-truth point {idx}", CAP_COL, font=font_large)
                        blank_video_gen[iframe] = np.array(tmp)
                    plotted.append(blank_video_gen[:, ::3, ::3, :])

                blank_video_gen = np.concatenate([
                    np.concatenate([plotted[0], plotted[1], plotted[2]], axis=2),
                    np.concatenate([plotted[3], plotted[4], plotted[5]], axis=2),
                    np.concatenate([plotted[6], plotted[7], plotted[8]], axis=2),
                ], axis=1)
                blank_video_container.append(blank_video_gen)
    blank_video = np.concatenate(blank_video_container, axis=0)
    ffmpeg_video_write(blank_video, video_path, fps=15)

def spatial_align(input_video, reconstructed_video):
    # [num_snippets, LENGTH*17*3]
    num_snippets = input_video.shape[0]
    assert reconstructed_video.shape[0] == num_snippets
    ttl = input_video.shape[1]
    aligned = []
    for i in range(num_snippets):
        x = input_video[i]
        body_centre_x_to = (x[33] + x[36]) / 2
        body_centre_y_to = (x[34] + x[37]) / 2
        body_centre_z_to = (x[35] + x[38]) / 2
        y = reconstructed_video[i]
        body_centre_x_fm = (y[33] + y[36]) / 2
        body_centre_y_fm = (y[34] + y[37]) / 2
        body_centre_z_fm = (y[35] + y[38]) / 2
        shift = np.tile(np.array([body_centre_x_to - body_centre_x_fm,
                                  body_centre_y_to - body_centre_y_fm,
                                  body_centre_z_to - body_centre_z_fm]), ttl // 3)
        aligned.append(y + shift)
    return np.vstack(aligned)

def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B)
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))
    t = -np.dot(R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return R, t

def rigid_align(A, B):
    # both numpy array of [J, 3], align A to B
    R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(R, np.transpose(A))) + t
    return A2

def rigid_align_sequence(A, B):
    # align A's first frame to B
    # A, [T1, J, 3]
    # B, [T2, J, 3]
    R, t = rigid_transform_3D(A[0], B[-1])
    a_container = []
    for a in A:
        a2 = np.transpose(np.dot(R, np.transpose(a))) + t
        a_container.append(a2)
    A2 = np.stack(a_container, axis=0)
    return A2
