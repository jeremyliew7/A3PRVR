# -*- coding: utf-8 -*-
"""Verify published AO8RW feature lengths; this is not a feature extractor."""

from pathlib import Path
import sys

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from method.data_provider import uniform_feature_sampling

feature_file = (
    PROJECT_ROOT
    / "dataset/charades_i3d/FeatureData/charades_clip_i3d_numpy.hdf5"
)
video_id = "AO8RW"
max_len = 128  # Official default max_ctx_l; an upper limit, not a target length.

# Read the real arrays from the author's published feature file, read-only.
with h5py.File(feature_file, "r") as features:
    i3d = features[video_id]["i3d_feat"][...]
    clip = features[video_id]["clip_feat"][...]

print("video_id:", video_id)
print("I3D before sampling:", i3d.shape)
print("CLIP before sampling:", clip.shape)

# Run the same sampling function used by Dataset4MS_SL.__getitem__.
i3d_sampled = uniform_feature_sampling(i3d, max_len)
clip_sampled = uniform_feature_sampling(clip, max_len)

print("I3D after sampling:", i3d_sampled.shape)
print("CLIP after sampling:", clip_sampled.shape)
print("Returned original arrays:", i3d_sampled is i3d, clip_sampled is clip)

assert i3d.shape == (101, 1024)
assert clip.shape == (102, 512)
assert i3d_sampled is i3d and clip_sampled is clip

print("Verified: 101/102 already exist in the published feature file.")
