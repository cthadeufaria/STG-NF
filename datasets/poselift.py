"""
PoseLift dataset loader for STG-NF.

Reads pickle files from PoseLift/Pickle_files/{Train,Test}/ and produces
sliding-window pose segments compatible with PoseSegDataset's interface.

Pickle format (per file e.g. "1_126.pkl"):
    {frame_num: {person_id: [bbox(4,), keypoints(17,3)]}}
    keypoints columns: [x_pixel, y_pixel, confidence]

Output per item: [data, trans_index, score, label]
    data:        float32 tensor [3, seg_len, 18]   (C=x/y/conf, T=frames, V=joints)
    trans_index: int 0
    score:       float32 tensor [seg_len]           (per-frame mean keypoint confidence)
    label:       float32 scalar                     (1.0 = normal)

.metadata: np.ndarray [N, 4] — [scene_id, clip_id, person_id, start_frame]
"""

import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

# Import STG-NF utilities via sys.path (stg_nf_official/ is the working dir during training)
from dataset import keypoints17_to_coco18
from utils.data_utils import normalize_pose

# PoseLift video resolution — pixel coords observed max ~972 x ~1384
POSELIFT_VID_RES = [1024, 1440]


def _build_segments(pickle_dir, seg_len, stride):
    """
    Load all pickle files in pickle_dir and build sliding-window segments.

    Returns
    -------
    segs_data : np.ndarray  [N, seg_len, 18, 3]  — un-normalised COCO18 keypoints
    segs_meta : list of [scene_id, clip_id, person_id, start_frame]
    segs_score: np.ndarray  [N, seg_len]          — per-frame mean confidence
    """
    segs_data = []
    segs_meta = []
    segs_score = []

    for fname in sorted(os.listdir(pickle_dir)):
        if not fname.endswith(".pkl"):
            continue
        stem = os.path.splitext(fname)[0]
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        scene_id, clip_id = int(parts[0]), int(parts[1])

        with open(os.path.join(pickle_dir, fname), "rb") as fh:
            clip_data = pickle.load(fh)  # {frame_num: {person_id: [bbox, kp(17,3)]}}

        # Group frames by person
        tracks = {}  # {person_id: {frame_num: kp(17,3)}}
        for frame_num, persons in clip_data.items():
            for person_id, person_data in persons.items():
                kp = np.array(person_data[1], dtype=np.float32)  # (17, 3)
                if kp.shape != (17, 3):
                    continue
                if person_id not in tracks:
                    tracks[person_id] = {}
                tracks[person_id][frame_num] = kp

        for person_id, track in tracks.items():
            frame_nums = sorted(track.keys())
            if len(frame_nums) < seg_len:
                continue

            # Stack frames: [T, 17, 3]
            track_kps = np.stack([track[f] for f in frame_nums])

            # COCO17 → COCO18: [T, 18, 3]
            track_kps18 = keypoints17_to_coco18(track_kps)

            T = len(frame_nums)
            for start in range(0, T - seg_len + 1, stride):
                seg = track_kps18[start : start + seg_len]  # [seg_len, 18, 3]
                per_frame_conf = seg[:, :, 2].mean(axis=1)  # [seg_len]

                segs_data.append(seg)
                segs_meta.append([scene_id, clip_id, person_id, frame_nums[start]])
                segs_score.append(per_frame_conf)

    if not segs_data:
        raise RuntimeError(f"No segments found in {pickle_dir}")

    return (
        np.stack(segs_data, axis=0),   # [N, seg_len, 18, 3]
        segs_meta,
        np.stack(segs_score, axis=0),  # [N, seg_len]
    )


class PoseLiftDataset(Dataset):
    """
    PoseLift dataset compatible with PoseSegDataset interface.

    Interface contract:
        __getitem__  → [data_tensor, trans_index, score_tensor, label_tensor]
        .metadata    → np.ndarray [N, 4]
    """

    def __init__(self, pickle_dir, seg_len=24, stride=1, vid_res=None):
        vid_res = vid_res or POSELIFT_VID_RES

        raw_data, meta_list, score_np = _build_segments(pickle_dir, seg_len, stride)
        # raw_data: [N, seg_len, 18, 3]

        # Normalise: divide by vid_res, zero-mean over sequence, /std_y
        # normalize_pose expects [B, T, V, C] → returns same shape
        norm_data = normalize_pose(raw_data, vid_res=vid_res)  # [N, seg_len, 18, 3]

        # Transpose to [N, C, T, V] = [N, 3, seg_len, 18]
        self.segs_data_np = norm_data.transpose(0, 3, 1, 2).astype(np.float32)

        self.metadata = np.array(meta_list)                         # [N, 4]
        self.segs_score_np = score_np.astype(np.float32)           # [N, seg_len]
        self.labels = np.ones(len(meta_list), dtype=np.float32)    # all normal

    def __len__(self):
        return len(self.segs_data_np)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.segs_data_np[idx])        # [3, seg_len, 18]
        score = torch.from_numpy(self.segs_score_np[idx])      # [seg_len]
        label = torch.tensor(self.labels[idx])
        return [data, 0, score, label]
