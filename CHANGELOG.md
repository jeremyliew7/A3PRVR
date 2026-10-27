# A3PRVR Server Reproduction Changelog

> Scope: server workspace audit for the checkpoint-evaluation reproduction on 2026-09-03. This is not an upstream release changelog.

## Baseline

- Upstream remote: https://github.com/chuanshen-chen/A3PRVR.git
- Baseline revision: `ae02a70f7d6d5320531df0113ee48e47712a072d`
- Repository path: `~/A3PRVR`

## Source-code changes relative to upstream

**None.** Before this file was added, both `git diff` and `git diff --cached` were empty. The tracked source files, including `method/` and all three official evaluation scripts, were not edited.

In particular, `do_tvr.sh`, `do_activitynet.sh`, and `do_charades.sh` retain their upstream contents. No model architecture, loss, optimizer, data-loader, metric, or checkpoint-loading code was changed.

## Runtime evaluation differences from the upstream scripts

These were command-line choices for a one-pass checkpoint evaluation only; they were not saved into source code.

| Dataset | Upstream-script behavior | Actual evaluation behavior | Effect |
|---|---|---|---|
| TVR | `do_tvr.sh` runs one command with the parser default `n_epoch=100`. | Ran the unmodified script. It therefore evaluated the same checkpoint 100 times. | No parameter override; redundant evaluations only. |
| ActivityNet-Caption | `do_activitynet.sh` loops 10 times; each command otherwise inherits `n_epoch=100`. Its description begins with `remake_anet_wo_pos_173.2`. | Ran one equivalent command with `--n_epoch 1` and description `repro_activitynet_checkpoint_eval`. | Avoided 1,000 redundant evaluation passes. `n_epoch` only bounds the evaluation loop; description only changes the output-directory name. |
| Charades-STA | `do_charades.sh` inherits `n_epoch=100` and uses description `charades`. | Ran one equivalent command with `--n_epoch 1` and description `repro_charades_checkpoint_eval`. | Avoided 100 redundant evaluation passes. `n_epoch` only bounds the evaluation loop; description only changes the output-directory name. |

All model- and dataset-relevant arguments matched the respective upstream script:

- `--only_eval` with the released checkpoint: `tvr.ckpt`, `anet.ckpt`, or `charades.ckpt`.
- I3D features, RoBERTa query features, `--best_frame_loss`, `--phrase_action_branch`, `--flow_feat --visual_flow_feat_dim 512`, batch size 128, and cross-branch fusion.
- Q-DTA: 8 heads, 8 offset groups, 8 offsets; scale 64 for TVR and 96 for ActivityNet-Caption / Charades-STA.
- Seeds: TVR 808, ActivityNet-Caption 520, Charades-STA 666.

## Environment and data additions (not source-code edits)

- Created the user-local Conda environment `~/miniconda3/envs/a3prvr` with Python 3.9 and PyTorch 2.0.1 + CUDA 11.7.
- Downloaded the released Charades-STA, ActivityNet-Caption, and TVR data plus the three released checkpoints into ignored `dataset/` and `checkpoints/` directories. Download locations were renamed to the paths expected by the unmodified code: `charades_i3d`, `activitynet_i3d`, and `tvr_i3d`.
- Deleted only the incomplete checkpoint download cache `checkpoints/.cache` after validating the three payload files; released checkpoint files were retained.
- Generated ignored `results_aaai26/` evaluation outputs and untracked download/evaluation log files. These are artifacts, not source modifications.

## Checkpoint evaluation results

| Dataset | R@1 | R@5 | R@10 | R@100 | SumR | Paper SumR |
|---|---:|---:|---:|---:|---:|---:|
| Charades-STA | 2.9 | 9.6 | 15.5 | 54.3 | 82.3 | 82.3 |
| ActivityNet-Caption | 11.8 | 32.7 | 46.1 | 82.3 | 173.0 | 173.2 |
| TVR | 27.0 | 52.5 | 63.6 | 92.3 | 235.4 | 235.3 |

Root logs:

- `repro-a3prvr-charades.log`
- `repro-a3prvr-activitynet.log`
- `repro-a3prvr-tvr.log`

## Verification

- The data payload sizes were checked against the Hugging Face release manifest before evaluation.
- The final metrics above are the fused-branch metrics in the corresponding root log.
- No training was run; all reported values are released-checkpoint evaluations.
