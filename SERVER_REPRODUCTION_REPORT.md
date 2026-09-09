---
title: A3PRVR 服务器 checkpoint 评测复现报告
type: reproduction-report
date: 2026-09-03
scope: released-checkpoint evaluation
---

# A3PRVR 服务器 checkpoint 评测复现报告

## 结论

在学校服务器 iipl-100 上，使用作者发布的 checkpoint、发布特征和官方代码提交 `ae02a70f7d6d5320531df0113ee48e47712a072d` 完成了 Charades-STA、ActivityNet-Caption 和 TVR 的评测。Charades-STA 与论文 Table 1 完全一致；ActivityNet-Caption 的 SumR 低 0.2，TVR 的 SumR 高 0.1，均为很小的数值差异。

本报告只记录 **checkpoint 评测复现**，不表示已从头训练复现论文。

| 数据集 | R@1 | R@5 | R@10 | R@100 | SumR | mAP | 论文 SumR | 差异 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Charades-STA | 2.9 | 9.6 | 15.5 | 54.3 | 82.3 | 0.0746 | 82.3 | 0.0 |
| ActivityNet-Caption | 11.8 | 32.7 | 46.1 | 82.3 | 173.0 | 0.2252 | 173.2 | -0.2 |
| TVR | 27.0 | 52.5 | 63.6 | 92.3 | 235.4 | 0.3908 | 235.3 | +0.1 |

论文的对应完整 R@K 为：Charades-STA 2.9/9.6/15.5/54.3，ActivityNet-Caption 11.8/32.9/46.2/82.3，TVR 27.0/52.4/63.6/92.3。

## 已验证的服务器基线

- 代码：`https://github.com/chuanshen-chen/A3PRVR.git`，提交 `ae02a70f7d6d5320531df0113ee48e47712a072d`。
- 路径：`~/A3PRVR`。
- GPU：NVIDIA GeForce RTX 3080 Ti（12 GiB；本次评测使用 GPU 0）。
- 环境：`~/miniconda3/envs/a3prvr`，Python 3.9.25、PyTorch 2.0.1+cu117、torchvision 0.15.2+cu117、Transformers 4.46.1。
- 数据与三个 checkpoint 的逐文件字节数均已与 Hugging Face 发布清单核验。
- 评测前 `git diff` 和 `git diff --cached` 均为空：未修改模型、损失、指标、数据加载或官方 `do_*.sh` 脚本。

## 自己复现：从零准备服务器

以下命令假设当前目录是服务器用户目录，且已安装 Miniconda。建议使用 Linux shell；本项目的 shell 脚本不适用于 Windows PowerShell。

```bash
git clone https://github.com/chuanshen-chen/A3PRVR.git ~/A3PRVR
cd ~/A3PRVR
git checkout ae02a70f7d6d5320531df0113ee48e47712a072d

~/miniconda3/bin/conda create -y -n a3prvr python=3.9
~/miniconda3/bin/conda run -n a3prvr pip install \
  torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
  --index-url https://download.pytorch.org/whl/cu117
~/miniconda3/bin/conda run -n a3prvr pip install \
  numpy==1.24.4 transformers==4.46.1 h5py easydict einops tqdm timm tensorboard \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`requirements.txt` 中的依赖行目前全被注释，因此不能只依赖 `pip install -r requirements.txt` 完成环境安装。

环境验收：

```bash
cd ~/A3PRVR
~/miniconda3/bin/conda run -n a3prvr python -c \
  'import torch, transformers, h5py, easydict, einops, timm; print(torch.__version__); print(torch.cuda.is_available())'
~/miniconda3/bin/conda run -n a3prvr python method/train.py --help
```

预期看到 `2.0.1+cu117` 和 `True`。偶发的 NVML 初始化警告不影响 PyTorch 使用 CUDA。

## 下载数据和 checkpoint

发布仓库是 `chadcs/A3PRVR_AAAI2026`。服务器在国内网络下可使用 Hugging Face 镜像；禁用 Xet 可规避已遇到过的 Xet CAS 兼容问题。

```bash
cd ~/A3PRVR
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# 先保证 hf 命令可用；未安装时执行：python -m pip install --user -U huggingface_hub
hf download chadcs/A3PRVR_AAAI2026 --include 'charades_dataset/**' --local-dir .hf-staging
hf download chadcs/A3PRVR_AAAI2026 --include 'anet_dataset/**'     --local-dir .hf-staging
hf download chadcs/A3PRVR_AAAI2026 --include 'tvr_dataset/**'      --local-dir .hf-staging
hf download chadcs/A3PRVR_AAAI2026 --include '*.ckpt'              --local-dir .hf-staging

mkdir -p dataset checkpoints
mv .hf-staging/charades_dataset dataset/charades_i3d
mv .hf-staging/anet_dataset     dataset/activitynet_i3d
mv .hf-staging/tvr_dataset      dataset/tvr_i3d
mv .hf-staging/charades.ckpt .hf-staging/anet.ckpt .hf-staging/tvr.ckpt checkpoints/
```

如只复现论文 README 推荐的 Charades-STA，可只下载 `charades_dataset/**` 和 `charades.ckpt`。其发布文件与已验证字节数如下：

| 相对路径 | 字节数 |
|---|---:|
| `FeatureData/charades_clip_i3d_numpy.hdf5` | 5,636,823,728 |
| `TextData/roberta_charades_query_feat.hdf5` | 559,365,376 |
| `TextData/charadestrain.caption.txt` | 557,378 |
| `TextData/charadesval.caption.txt` | 167,562 |
| `TextData/charades_id_action_dict.json` | 664,586 |
| `TextData/charades_id_object_dict.json` | 714,148 |
| `TextData/charades_total_action_dict.json` | 7,330 |
| `TextData/charades_total_object_dict.json` | 8,412 |
| `checkpoints/charades.ckpt` | 1,480,510,669 |

## 执行 checkpoint 评测

所有命令在 `~/A3PRVR` 执行，并使用发布 checkpoint。日志中会依次出现两个分支和融合分支的指标；应记录最后一组融合分支指标。

### Charades-STA：单次、高效评测

```bash
cd ~/A3PRVR
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/bin/conda run -n a3prvr python method/train.py \
  --dataset_name charades --visual_feature i3d --root_path . --dset_name charades \
  --exp_id debug --device_ids 0 --topk 0.2 \
  --caption_train_txt ./dataset/charades_i3d/TextData/charadestrain.caption.txt \
  --caption_test_txt ./dataset/charades_i3d/TextData/charadesval.caption.txt \
  --best_frame_loss --phrase_action_branch --flow_feat --visual_flow_feat_dim 512 --bsz 128 \
  --text_feat_path ./dataset/charades_i3d/TextData/roberta_charades_query_feat.hdf5 \
  --seed 666 --cross_branch_fusion --deformable_attn --deformable_heads 8 \
  --deformable_offset_groups 8 --deformable_offset_num 8 --deformable_offset_scale 96 \
  --only_eval --eval_ckpt checkpoints/charades.ckpt --n_epoch 1 \
  --description repro_charades_checkpoint_eval |& tee repro-a3prvr-charades.log
```

预期融合结果：`[2.9, 9.6, 15.5, 54.3]`，SumR `82.3`。

### ActivityNet-Caption：单次、高效评测

```bash
cd ~/A3PRVR
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/bin/conda run -n a3prvr python method/train.py \
  --dataset_name activitynet --visual_feature i3d --root_path . --dset_name activitynet \
  --exp_id aaai26_bestresult_forbetter_173.2 --device_ids 0 --topk 0.2 \
  --caption_train_txt ./dataset/activitynet_i3d/TextData/activitynettrain.caption.txt \
  --caption_test_txt ./dataset/activitynet_i3d/TextData/activitynetval.caption.txt \
  --best_frame_loss --phrase_action_branch --flow_feat --visual_flow_feat_dim 512 --bsz 128 \
  --text_feat_path ./dataset/activitynet_i3d/TextData/roberta_activitynet_query_feat.hdf5 \
  --seed 520 --cross_branch_fusion --deformable_attn --deformable_heads 8 \
  --deformable_offset_groups 8 --deformable_offset_num 8 --deformable_offset_scale 96 \
  --only_eval --eval_ckpt checkpoints/anet.ckpt --n_epoch 1 \
  --description repro_activitynet_checkpoint_eval |& tee repro-a3prvr-activitynet.log
```

预期融合结果：`[11.8, 32.7, 46.1, 82.3]`，SumR `173.0`。

### TVR：与本报告相同的官方脚本运行方式

```bash
cd ~/A3PRVR
~/miniconda3/bin/conda run -n a3prvr bash do_tvr.sh |& tee repro-a3prvr-tvr.log
```

预期融合结果：`[27.0, 52.5, 63.6, 92.3]`，SumR `235.4`。

`do_tvr.sh` 在 `--only_eval` 下没有显式设置 `n_epoch`，会使用默认的 100 并重复同一评测；这不会更新参数，但会耗时约 1 小时 50 分。若只需一次确定性核验，可将脚本最后的训练命令复制出来并追加 `--n_epoch 1`；本报告的 TVR 数字来自上面的未修改官方脚本。

## 与官方脚本的差异及其影响

- 三个官方 `do_*.sh` 脚本均未改动。
- TVR 使用未修改的 `do_tvr.sh`。
- ActivityNet-Caption 官方脚本包含 10 次外层循环，且默认 `n_epoch=100`；本次仅执行一条等价命令并传入 `--n_epoch 1`。
- Charades-STA 官方脚本默认 `n_epoch=100`；本次传入 `--n_epoch 1`。
- 上述 `n_epoch` 改动只限制 evaluation-only 的重复次数，未改变 checkpoint、模型、特征、损失或评测指标。两次单次评测还使用了新的 `description`，仅改变输出目录名称。

## 服务器证据与产物

- 根目录审计：`~/A3PRVR/CHANGELOG.md`。
- 原始评测日志：`~/A3PRVR/repro-a3prvr-charades.log`、`~/A3PRVR/repro-a3prvr-activitynet.log`、`~/A3PRVR/repro-a3prvr-tvr.log`。
- 生成的评测目录：`~/A3PRVR/results_aaai26/`。
- 本地论文对照：`20-Areas/research/papers/002-Action-and-object Aware Alignment for Partially Relevant Video Retrieval/pdfs/A3PRVR-AAAI2026.md` 的 Table 1。

## 边界与下一步

本结果证明发布 checkpoint 能在已验证环境中复现论文表格附近的性能；它没有验证随机初始化训练、早停点选择或训练吞吐。若要完成严格的从头训练复现，应单独记录随机种子、训练时长、每轮验证曲线、最优 checkpoint 和与此报告分开的训练日志。
