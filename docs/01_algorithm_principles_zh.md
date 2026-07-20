# ChemFlow3D 算法原理：从 Potential Flow 到 3D 点云 Latent Traversal

本文档凝练整理项目依赖的核心算法思想。主线是：以 ChemFlow-style potential latent flow 为实现框架，结合两篇关联工作 PoFlow 与 Flow Factorized Representation Learning 的理论视角，解释为什么可以在 3D 点云 AE latent space 中学习可控 traversal。

![framework](../chemflow3d_runs/final_figures/fig_01_framework.png)

## 1. 统一问题设定

给定点云样本：

```text
x ∈ R^{N×3}
```

使用编码器和解码器：

```text
z = E(x),      x̂ = D(z)
```

其中 `z ∈ R^d` 是连续 latent representation。我们不直接在点云空间移动 `x`，而是在 latent space 中学习一个随时间变化的 velocity field：

```text
dz(t) / dt = v(z(t), t)
```

离散化后：

```text
z_{t+1} = z_t + η v(z_t,t)
```

最终通过 decoder 得到点云轨迹：

```text
x_t = D(z_t)
```

本项目采用 ChemFlow / PoFlow 风格的势能梯度场：

```text
uθ(z,t): R^d × R → R
vθ(z,t) = ∇z uθ(z,t)
```

也就是说，模型不直接输出 `d` 维向量，而是输出一个 scalar potential，再通过自动微分得到方向。

## 2. PoFlow：Latent Traversals as Potential Flows

PoFlow 的核心思想是：在一个固定的生成模型或 autoencoder latent space 中，学习多个可解释 traversal direction。每个方向由一个势能函数产生：

```text
v_k(z,t) = ∇z u_k(z,t)
```

离散 traversal：

```text
z_{n+1} = z_n + α normalize(∇z u_k(z_n,t_n))
```

或者不归一化：

```text
z_{n+1} = z_n + α ∇z u_k(z_n,t_n)
```

### 2.1 PDE regularization

PoFlow 通过 PDE 约束让 potential function 更平滑、更结构化。典型形式是 Wave PDE：

```text
u_tt - c² Δz u = 0
```

其中：

```text
Δz u = tr(∇²_z u)
```

训练时使用 residual：

```text
L_wave = E[(u_tt - c² Δz u)²]
```

直观含义：

- `u(z,t)` 不是任意变化；
- 它被约束成类似“波”在 latent space 中传播；
- 这会抑制局部剧烈、不连续的 traversal。

### 2.2 Guidance signal

仅优化 PDE 会得到平凡解，因此需要下游指导信号。若属性函数为：

```text
p(D(z))
```

希望属性上升，则最大化方向导数：

```text
∇z p(D(z))ᵀ ∇z u(z,t)
```

在代码中对应：

```text
directional_guidance = <grad_property, velocity>
```

对分类目标，如果目标类别 logit 为：

```text
f_c(D(z))
```

则使用：

```text
∇z f_c(D(z))ᵀ ∇z u(z,t)
```

这就是 E2 class-target traversal 的数学基础。

## 3. Flow Factorized Representation Learning

Flow Factorized 的核心关注点不是单个 traversal，而是表示空间是否可以被分解为多个相对独立的动态因素。

给定 sequence：

```text
x_0, x_1, ..., x_T
```

编码后：

```text
z_t = E(x_t)
```

希望 latent dynamics 可以分解为：

```text
dz/dt = Σ_k a_k(t) v_k(z,t)
```

其中不同 `v_k` 对应不同变化因素，例如姿态、尺度、位置、形状属性等。

### 3.1 与本项目的关系

E3 synthetic sequence 实验正是这个思想的简化版本：

```text
x_t = G_{α_t}(x_0)
z_t = E(x_t)
v_gt = (z_{t+1} - z_t) / step_size
```

我们用 ground-truth 变换序列显式构造 latent displacement，并训练：

```text
∇z uθ(z_t,t) ≈ v_gt
```

因此 E3 回答的是：

> 对一个已知几何因素，例如 yaw 或 scale_x，AE latent space 中是否存在可学习、可 rollout 的方向场？

## 4. ChemFlow-style 训练目标

本项目的通用 flow objective 可以写成：

```text
L = λ_guide L_guide
  + λ_pde L_pde
  + λ_ic L_ic
  + λ_aux L_aux
```

不同实验使用不同 guidance：

| 实验 | guidance | 数学形式 |
|---|---|---|
| E1 | geometry property | `∇p(D(z))ᵀ∇u` |
| E2 | class target logit | `∇f_c(D(z))ᵀ∇u` |
| E3 | ground-truth latent velocity | `||∇u - v_gt||² + 1-cos(∇u,v_gt)` |

PDE 选项：

```text
wave: u_tt - c² Δu
hj:   u_t + 1/2 ||∇u||²
none: no PDE residual
```

其中 Hamilton-Jacobi residual：

```text
L_hj = E[(u_t + 1/2 ||∇z u||²)²]
```

## 5. 为什么适合 3D 点云？

3D 点云存在几个适合 latent flow 的特点：

1. 点云空间高维、无序，直接编辑困难；
2. AE latent space 提供连续低维表示；
3. 几何属性、类别目标和 synthetic transform 都可以定义为 direction signal；
4. 解码器可以把 latent trajectory 投影回点云形状；
5. E3 可以构造有 ground truth 的几何变换，用来检验是否真的学到方向。

## 6. 关键限制：势能梯度场的无旋约束

由于：

```text
v(z,t) = ∇z u(z,t)
```

固定时间切片上，velocity field 的 Jacobian 是 Hessian：

```text
∂v/∂z = ∇²_z u
```

Hessian 是对称矩阵，因此该场天然缺少局部旋转分量。直观地说，单一 scalar potential gradient 更适合表达：

- 单调拉伸；
- 缩放；
- 属性上升；
- 朝某个类别决策区域移动。

但它不一定适合表达：

- 闭合环流；
- 纯旋转；
- 局部 curl-dominated dynamics。

这正好解释了 E3 中的现象：

```text
scale_x direction cosine ≈ 0.94
yaw direction cosine     ≈ 0.75
```

scale_x 是单调伸缩，更符合 conservative potential field；yaw 更接近旋转/环流，因此更难。

## 7. 本项目的算法贡献点

当前 ChemFlow3D 项目形成了一个清晰实验框架：

1. 将 ChemFlow/PoFlow 的 potential flow 迁移到 3D point-cloud latent space；
2. 同时覆盖 property、class、ground-truth synthetic dynamics 三类任务；
3. 用 E3 明确区分“latent direction 学到了”和“decoded geometry 是否真实”；
4. 用 yaw vs scale_x 对比初步揭示 potential flow 的无旋限制；

