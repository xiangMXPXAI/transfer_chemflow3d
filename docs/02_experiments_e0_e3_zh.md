# ChemFlow3D 实验报告：E0–E3

本文档汇总 ChemFlow 迁移到 3D 点云后的主线实验。实验对象为 ModelNet10 点云，核心目标是验证：自编码器潜空间是否可用，ChemFlow-style latent flow 是否能控制几何属性、类别方向，以及在有 ground truth 的合成变换中是否真正学习了几何方向。

![E0-E3 dashboard](../chemflow3d_runs/final_figures/fig_02_e0_e3_dashboard.png)

![3D point cloud rollouts](../chemflow3d_runs/final_figures/fig_03_e3_3d_pointcloud_rollouts.png)

## 1. 实验链路

整体实验被拆成四个阶段。

| 阶段 | 问题 | 核心验证 |
|---|---|---|
| E0 | 点云 AE 与分类器基线是否可靠 | 潜空间是否保留几何与语义信息 |
| E1 | 几何属性 latent flow 是否有效 | height / width / volume 是否沿指定方向单调变化 |
| E2 | 类别目标 latent flow 是否有效 | 是否能把样本推向目标类别 2 / 8 |
| E3 | flow 是否学习真实几何方向 | yaw 与 scale_x 合成序列中，预测轨迹是否接近 ground truth |

数据流如下：

```text
ModelNet10 点云
  -> PointNet-style Encoder
  -> latent z
  -> scalar potential / vector field
  -> latent ODE rollout
  -> Decoder 重建点云
  -> 几何属性、分类器、GT 序列指标评估
```

实验中比较三类方向：

| 方法 | 含义 | 作用 |
|---|---|---|
| Random | 随机单位 latent 方向 | 检查任务不是由任意扰动完成 |
| Direct gradient | 直接使用属性/类别梯度 | 近似上界或教师方向 |
| PDE latent flow | 学习势函数并加入 no-pde / Wave / HJ 约束 | 项目主方法 |

## 2. E0：AE 与分类器基线

E0 的任务是建立可用潜空间。如果 AE 重建质量差，后续 latent flow 的点云可视化与几何评估都不可信。

| 指标 | 数值 |
|---|---:|
| 测试样本数 | 908 |
| mean Chamfer | 0.00544 |
| median Chamfer | 0.00489 |
| p90 Chamfer | 0.00883 |
| 原始点云分类准确率 | 91.30% |
| 重建点云分类准确率 | 90.64% |

结论：E0 通过。AE 在 ModelNet10 上保持了较好的几何重建质量，重建后分类准确率仅比原始点云低约 0.66 个百分点，说明潜空间对后续几何与类别 flow 实验是可用的。

## 3. E1：几何属性方向

E1 针对 height、width、volume 三个显式几何属性训练 latent flow。评估时从测试样本 latent code 出发，沿 learned flow rollout，并在解码点云上重新计算属性变化。

| 属性 | 方法 | mean Δ | positive rate |
|---|---|---:|---:|
| height | Random | 0.0039 | 55.18% |
| height | Wave | 1.4319 | 100.00% |
| height | HJ | 1.4294 | 100.00% |
| width | Random | 0.0066 | 57.60% |
| width | Wave | 1.1919 | 100.00% |
| width | HJ | 1.1909 | 100.00% |
| volume | Random | 0.0187 | 57.49% |
| volume | Wave | 5.1945 | 100.00% |
| volume | HJ | 5.1254 | 100.00% |

现象：

- Random 方向几乎不改变属性，positive rate 只略高于 50%，符合随机扰动预期。
- Wave 与 HJ 在三个属性上都达到 100% 正向变化，说明模型确实学习到了属性上升方向。
- Wave 与 HJ 差异很小，说明在当前 E1 设定下，主要贡献来自属性梯度监督，PDE 正则带来的差异不是决定性因素。

结论：E1 通过。ChemFlow-style flow 能在点云 latent space 中稳定控制显式几何属性。

## 4. E2：目标类别方向

E2 将 guidance 从几何属性换成分类器目标 logit / margin。实验分别训练 target_class=2 与 target_class=8。需要注意：这里没有固定单一 source_class，而是从测试集中排除目标类别样本后，使用所有非目标类别作为 source distribution。因此 E2 验证的是“多源类别到目标类别”的迁移，而不是“一对一类别转换”。

| target | 方法 | initial success | final success | new success | positive margin |
|---:|---|---:|---:|---:|---:|
| 2 | Random | 0.62% | 0.74% | 0.25% | 59.03% |
| 2 | Direct class gradient | 0.62% | 91.58% | 90.97% | 100.00% |
| 2 | Wave | 0.62% | 89.85% | 89.23% | 100.00% |
| 2 | HJ | 0.62% | 91.09% | 90.47% | 100.00% |
| 8 | Random | 1.36% | 1.24% | 0.37% | 55.32% |
| 8 | Direct class gradient | 1.36% | 57.92% | 56.56% | 99.50% |
| 8 | Wave | 1.36% | 53.34% | 52.23% | 95.54% |
| 8 | HJ | 1.36% | 53.96% | 52.72% | 96.04% |

现象：

- target=2 的成功率接近直接类别梯度上界，说明该目标类别在 latent space 中有较清晰的分类边界方向。
- target=8 的成功率明显低于 target=2，但仍显著高于 Random，说明模型学到了有效类别方向，但目标类别可达性更弱。
- HJ 在 target=2 与 target=8 上均略高于 Wave，但差距不大。
- E2 的点云可视化应谨慎解释：类别迁移不是严格的配对几何变换，分类器成功不必然等价于人眼可见的完整形态转换。

结论：E2 基本通过。latent flow 能显著提升目标类别概率和目标类别命中率，但 target=8 暴露出类别方向受源分布、分类边界和 decoder 表达能力共同限制。

## 5. E3：有 ground truth 的几何方向

E3 是对 latent flow 最关键的诊断实验。我们构造合成序列，把每个点云做已知几何变换，并将序列编码到 latent space。模型需要学习从当前 latent 指向下一个 latent 的真实速度方向。

### 5.1 Yaw 旋转

| 方法 | endpoint Chamfer ↓ | trajectory Chamfer ↓ | velocity cosine ↑ | improvement over identity ↑ |
|---|---:|---:|---:|---:|
| Wave | 0.07379 | 0.03857 | 0.76248 | 97.03% |
| HJ | 0.04430 | 0.02592 | 0.75473 | 97.58% |
| no-pde | 0.03593 | 0.02360 | 0.74944 | 96.81% |

现象：

- 三种方法的 velocity cosine 都约为 0.75，positive cosine rate 接近 1，说明方向整体正确。
- no-pde 的 endpoint 与 trajectory Chamfer 最低，HJ 次之，Wave 最弱。
- yaw 是旋转型变换，其真实向量场天然带有环流结构；标量势能梯度场是无旋场，因此该任务正好暴露了 potential-flow 表达限制。

结论：E3-yaw 部分通过。模型学到了大体正确的 latent 方向，但势能梯度场对旋转型方向不够自然，PDE 正则没有带来稳定优势。

### 5.2 Scale-x 缩放

| 方法 | endpoint Chamfer ↓ | median endpoint ↓ | trajectory Chamfer ↓ | velocity cosine ↑ | positive cosine |
|---|---:|---:|---:|---:|---:|
| Wave | 0.02848 | 0.01339 | 0.01130 | 0.93835 | 100.00% |
| HJ | 0.02399 | 0.01319 | 0.01036 | 0.93826 | 100.00% |
| no-pde | 0.03002 | 0.01309 | 0.01143 | 0.93808 | 100.00% |

现象：

- velocity cosine 约为 0.938，positive cosine rate 达到 100%，明显优于 yaw。
- HJ 的 mean endpoint / trajectory 指标最好，no-pde 的 median endpoint 略好。
- scale_x 更接近可由势能梯度描述的伸缩方向，因此比 yaw 更适配当前框架。

结论：E3-scale_x 通过。latent flow 能稳定学习 ground-truth 几何方向，说明 E1/E2 的成功不是单纯的指标偶然，而是 latent space 中确实存在可学习的几何速度场。

## 6. 总体对比结论

综合 E0–E3，可以得到以下结论。

| 维度 | 判断 |
|---|---|
| 潜空间质量 | E0 通过，AE 与分类器足够支撑 flow 实验 |
| 属性控制 | E1 通过，height / width / volume 可稳定正向控制 |
| 类别控制 | E2 基本通过，target=2 强，target=8 中等 |
| 几何方向学习 | E3 通过，scale_x 明确成功，yaw 暴露势能场限制 |
| PDE 差异 | HJ 通常略稳，Wave 在 yaw 中偏弱，no-pde 是强基线 |
| 方法瓶颈 | 标量势函数产生无旋梯度场，难以表达旋转/环流型变换 |

最重要的实验结论不是“某个 PDE 全面最好”，而是：

1. ChemFlow 迁移到 3D 点云 latent space 是可行的；
2. 对伸缩、尺寸、体积这类梯度型几何变化，势能场方法表现稳定；
3. 对 yaw 这类旋转型变换，当前 potential-flow 框架存在表达瓶颈；
4. 后续如果要形成更强项目贡献，应重点突破“无旋限制”，而不是只继续堆叠更多 PDE loss。



