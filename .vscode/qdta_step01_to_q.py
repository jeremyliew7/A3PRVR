"""第一步：用小输入观察官方 Q-DTA 的 to_q 投影；CPU 学习用。"""

import torch

from method.deformable_module.deformable_1d_self import DeformableAttention1D


torch.manual_seed(7)

# 复用官方模块；本轮只调用其中的 to_q。
qdta = DeformableAttention1D(
    dim=8,
    heads=4,
    offset_groups=2,
    offset_num=2,
    downsample_factor=1,
    offset_kernel_size=3,
    offset_scale=2,
    dim_ff=16,
    position=False,
)
qdta.eval()

# 已编码动作特征的教学替身：[Batch, Time, Feature] = [1, 3, 8]。
# 对应伪代码中的 A；数值仅用于观察，不代表真实视频语义。
A = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 10.0

print("本轮观察：A -> 官方 to_q -> Q", flush=True)
print(qdta.to_q, flush=True)

with torch.no_grad():
    # 在下一行按 F9 设置断点；首次停住时 Q 尚未生成。
    Q = qdta.to_q(A)

print("A.shape:", tuple(A.shape))
print("Q.shape:", tuple(Q.shape))
print("A[0, 0]:", A[0, 0])
print("Q[0, 0]:", Q[0, 0])
