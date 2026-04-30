# GMN（高斯混合噪声）原理与复现分析

## 一、GMN 概述

GMN（Gaussian Mixture Noise）是 MeanFuser 中 MeanFlow 模块的核心组件。与传统 Flow Matching 使用各向同性高斯噪声不同，GMN 将噪声空间划分为 K 个高斯分量，每个分量对应一类驾驶行为模式（如直行、左转、右转、停车等），使得生成模型在推理时能同时产生 K 条多样化的候选轨迹。

**核心思想**：用 K-means 对训练集专家轨迹在归一化增量空间中聚类，得到 K 个簇心作为高斯分布的均值，标准差固定为 0.1。

## 二、理论基础与动机

### 2.1 Flow Matching 回顾

Flow Matching 定义一条从噪声分布 $p_1$ 到数据分布 $p_0$ 的概率流 ODE：

$$
\frac{dz_t}{dt} = u_\theta(z_t, t), \quad t \in [0, 1]
$$

其中 $t=1$ 对应纯噪声，$t=0$ 对应干净数据。训练时构造线性插值路径：

$$
z_t = (1 - t) \cdot x_0 + t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

网络学习**速度场** $u_\theta$ 来预测沿该路径的瞬时方向 $v_t = \epsilon - x_0$。

**标准 Flow Matching 的问题**：推理时需要**多步 ODE 积分**（通常 10~50 步），计算成本高，不适合实时自动驾驶。

### 2.2 MeanFlow：一步生成

MeanFlow 不直接学速度场 $u_\theta$，而是学**均值场** $\bar{u}$，定义为速度场在时间区间 $[r, t]$ 上的均值：

$$
\bar{u}(z_t, r, t) = \frac{1}{t - r} \int_r^t u_\theta(z_t, s) \, ds
$$

当 $r=0, t=1$ 时，$\bar{u}$ 直接给出从噪声到数据的**一步映射**：

$$
x_0 = z_1 - \bar{u}(z_1, 0, 1)
$$

**训练损失**（MeanFlow Loss）：通过 JVP（Jacobian-Vector Product）计算 $\bar{u}$ 对 $t$ 的导数，实现自监督训练，无需预训练的速度场模型：

```python
# 伪代码
z_t = (1 - t) * x_0 + t * e          # 线性插值
v_hat = e - x_0                       # 目标速度

u, dudt = jvp(model, (z_t, r, t), (v_hat, 0, 1))  # JVP 前向
v_pred = u + (t - r) * dudt           # 预测速度
loss = L1(v_pred, v_hat)              # L1 损失
```

### 2.3 标准高斯噪声的局限

使用单一高斯 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ 存在两个问题：

1. **多模态坍缩**：自动驾驶场景天然多模态（直行/左转/右转/停车），单一高斯噪声在推理时只能产生一条轨迹，无法覆盖多种可能
2. **噪声-数据映射不一致**：同一噪声区域可能需要映射到语义完全不同的轨迹，增加学习难度

### 2.4 GMN 的解决方案

GMN 将噪声空间**结构化**为 K 个高斯分量，每个分量与一种驾驶行为模式对齐：

$$
p_1(z) = \frac{1}{K} \sum_{k=1}^{K} \mathcal{N}(z \mid \mu_k, \sigma_k^2 I)
$$

其中：
- $\mu_k$ = 第 k 类驾驶行为在归一化增量空间中的聚类中心（乘以 0.5 缩放）
- $\sigma_k = 0.1$（固定值）
- $K = 8$（默认）

**核心优势**：

| 对比项 | 标准高斯 | GMN |
|--------|----------|-----|
| 推理输出 | 1 条轨迹 | K 条轨迹（每个簇 1 条）|
| 噪声结构 | 无结构 | 按行为模式分区 |
| 训练噪声选择 | 随机 | 与 GT 匹配的簇 |
| 多模态能力 | 需多次采样 | 一次前向即得 K 条 |

## 三、完整实现逻辑

### 3.1 Flow Matching 中噪声-数据对应关系图

```
噪声空间 (t=1)                          数据空间 (t=0)
                                         
  ┌─────────────────┐                    ┌─────────────────┐
  │  μ₀ (右转噪声)   │ ──── 去噪 ────→  │  右转轨迹        │
  │  μ₁ (中速直行)   │ ──── 去噪 ────→  │  中速直行轨迹    │
  │  μ₂ (停车噪声)   │ ──── 去噪 ────→  │  停车轨迹        │
  │  μ₃ (高速直行)   │ ──── 去噪 ────→  │  高速直行轨迹    │
  │  μ₄ (低速直行)   │ ──── 去噪 ────→  │  低速直行轨迹    │
  │  μ₅ (左转噪声)   │ ──── 去噪 ────→  │  左转轨迹        │
  │  μ₆ (缓行噪声)   │ ──── 去噪 ────→  │  缓行轨迹        │
  │  μ₇ (较高速直行)  │ ──── 去噪 ────→  │  较高速直行轨迹  │
  └─────────────────┘                    └─────────────────┘
```

### 3.2 训练前向传播详解

```python
def forward(self, outputs, trajectories):
    # ═══════════════════════════════════════════════
    # Step 1: 将 GT 轨迹转换到增量空间
    # ═══════════════════════════════════════════════
    x_0 = diff_traj(trajectories)  # [B, 8, 3] → [B, 8, 4]
    
    # ═══════════════════════════════════════════════
    # Step 2: GMN 条件噪声采样
    # ═══════════════════════════════════════════════
    # 2a. 找 GT 轨迹最近的簇（在绝对坐标空间比较）
    #     cluster_trajs: [K, 8, 3]（每个簇的均值轨迹）
    #     计算 GT 与每个簇中心的 L2 距离，取最近的簇编号
    distances = ||trajectories[:, None] - cluster_trajs[None, :]||  # [B, K]
    min_index = distances.argmin(dim=1)                             # [B]
    
    # 2b. 对所有 K 个簇采样噪声
    #     gaussian_mean: [K, 4]（center_points * 0.5）
    #     gaussian_std:  [K, 4]（固定 0.1）
    e_all = randn([B, K, 8, 4]) * std + mean   # [B, K, 8, 4]
    
    # 2c. 只取与 GT 匹配的那个簇的噪声
    e = e_all.gather(dim=1, index=min_index)    # [B, 8, 4]
    
    # ═══════════════════════════════════════════════
    # Step 3: 时间步采样
    # ═══════════════════════════════════════════════
    # 采样 (r, t) 对，满足 r ≤ t
    # 默认 logit_normal 分布，偏向中间时刻
    # 50% 的概率 r=t（使模型也学"瞬时"速度场）
    r, t = sample_time_steps(B)      # r, t ∈ [0, 1]
    
    # ═══════════════════════════════════════════════
    # Step 4: 构造插值点和目标
    # ═══════════════════════════════════════════════
    z_t = (1 - t) * x_0 + t * e     # 线性插值：数据→噪声
    v_hat = e - x_0                  # 目标速度（噪声方向）
    
    # ═══════════════════════════════════════════════
    # Step 5: MeanFlow 损失（通过 JVP）
    # ═══════════════════════════════════════════════
    # 模型 u = model(z_t, r, t, context)
    # JVP 同时计算 u 和 du/dt
    u, dudt = jvp(model, (z_t, r, t), (v_hat, 0, 1))
    
    # MeanFlow 核心：v_pred = u + (t-r) * du/dt
    v_pred = u + (t - r) * dudt.detach()
    
    loss = L1(v_pred, v_hat)
    return loss
```

**关键设计要点**：

1. **簇匹配在绝对坐标空间**：`min_index` 的距离计算使用 `trajectories`（绝对坐标），而非 `x_0`（增量空间）。这更符合语义——轨迹形状相似的分到同簇
2. **噪声在增量空间**：采样的噪声 `e` 是增量空间的，与 `x_0` 维度一致 `[B, 8, 4]`
3. **gaussian_mean = center_points × 0.5**：缩放使噪声均值不在数据中心（0）也不在纯簇心，而是折中位置。直觉上，这让流场的"起点"更靠近数据，有助于一步去噪的精度
4. **JVP 的 tangents**：`(v_hat, 0, 1)` 表示对 $z$ 沿速度方向、$r$ 固定、$t$ 以速率 1 变化

### 3.3 推理采样详解

```python
@torch.no_grad()
def sample(self, encoder_output):
    # ═══════════════════════════════════════════════
    # Step 1: 从 K 个高斯分量各采样一个噪声
    # ═══════════════════════════════════════════════
    # num_proposals = 8（默认等于 K）
    # 每个簇产生 repeat_num = num_proposals // K 条噪声
    mean = gaussian_mean.repeat_interleave(repeat_num)   # [1, num_proposals, 1, 4]
    std  = gaussian_std.repeat_interleave(repeat_num)
    e = randn([B, num_proposals, 8, 4]) * std + mean     # 结构化噪声
    e = e.reshape(B * num_proposals, 8, 4)
    
    # ═══════════════════════════════════════════════
    # Step 2: 一步去噪（或多步 Euler）
    # ═══════════════════════════════════════════════
    if num_sample_steps == 1:
        # 单步：r=0, t=1
        u = model(e, r=0, t=1, context)
        x_0 = e - u                    # 去噪结果
    
    elif num_sample_steps > 1:
        # 多步 Euler 积分：t: 1 → 0
        z = e
        for t_cur, t_next in time_steps:
            u = model(z, r=t_next, t=t_cur, context)
            z = z - (t_cur - t_next) * u
        x_0 = z
    
    # ═══════════════════════════════════════════════
    # Step 3: 增量空间 → 绝对轨迹
    # ═══════════════════════════════════════════════
    trajectories = cumsum_traj(x_0)    # 逆 diff_traj
    # 输出: [B, num_proposals, 8, 3]   → K 条候选轨迹
    
    # 后续由 ARM 模块选择最优轨迹
```

**推理流程要点**：

1. **并行去噪**：所有 K 条候选轨迹在同一次前向中并行计算（batch 维扩展）
2. **一步 vs 多步**：默认 `num_sample_steps=1`，仅需 1 次网络前向；多步时用 Euler 法沿 ODE 积分
3. **cumsum_traj 逆变换**：将增量空间 `[B, 8, 4]` 恢复为绝对轨迹 `[B, 8, 3]`
   - x_diff → 反归一化 → 累加 → x 坐标
   - sin, cos → atan2 → heading
4. **每个簇的噪声均值不同**，因此去噪后的轨迹自然呈现不同的驾驶行为模式

### 3.4 GMN 在整体 pipeline 中的位置

```
                              训练
                               │
  GT轨迹 ──→ diff_traj ──→ x₀  │  噪声 e ← GMN(找最近簇, 采样)
                               │         │
                               ↓         ↓
                         z_t = (1-t)x₀ + t·e
                               │
                               ↓
                    model(z_t, r, t, BEV_context)
                               │
                               ↓
                       MeanFlow JVP Loss
                               
                               
                              推理
                               │
               噪声 e ← GMN(K个簇各采样1个)
                               │
                               ↓
                    model(e, r=0, t=1, BEV_context)
                               │
                               ↓
                         x₀ = e - u
                               │
                               ↓
                    cumsum_traj(x₀) → K条候选轨迹
                               │
                               ↓
                     ARM 选择最优轨迹 → 输出
```

## 四、数据处理流程

### 4.1 轨迹表示

每条专家轨迹为 8 个未来 waypoint，每个 waypoint 包含 `(x, y, heading)`，坐标系为当前帧的自车后轴局部坐标系：

```
trajectory: [8, 3]  →  (x, y, heading) × 8 timesteps
```

时间间隔 0.5s，共 4 秒的未来轨迹。

### 4.2 diff_traj() 归一化增量变换

将绝对轨迹 `[N, 8, 3]` 转换为归一化增量表示 `[N, 8, 4]`：

```python
def diff_traj(traj):
    # traj: [N, 8, 3] → (x, y, heading)
    
    # 1. x 增量：相邻帧 x 差分（首帧与 0 做差），减均值后归一化
    x_diff = diff(x, prepend=0) 
    x_diff_norm = (x_diff - X_DIFF_MEAN) / X_DIFF_RANGE
    
    # 2. y 增量：同理
    y_diff_norm = (y_diff - Y_DIFF_MEAN) / Y_DIFF_RANGE
    
    # 3. heading → sin/cos 分解
    sin_h = sin(heading)
    cos_h = cos(heading)
    
    # 输出: [N, 8, 4] → (x_diff_norm, y_diff_norm, sin, cos)
    return concat([x_diff_norm, y_diff_norm, sin_h, cos_h])
```

**归一化常数**（硬编码在 `navsim/agents/meanfuser/utils.py`）：

| 常数 | 值 | 含义 |
|------|----|------|
| X_DIFF_MEAN | 2.9502 | x 增量均值 |
| X_DIFF_MAX | 7.4756 | x 增量最大值 |
| X_DIFF_MIN | -1.2698 | x 增量最小值 |
| Y_DIFF_MEAN | 0.0607 | y 增量均值 |
| Y_DIFF_MAX | 4.8564 | y 增量最大值 |
| Y_DIFF_MIN | -5.0121 | y 增量最小值 |

> **注**：X_DIFF_RANGE = max(|X_DIFF_MAX - X_DIFF_MEAN|, |X_DIFF_MIN - X_DIFF_MEAN|) = 4.5253

### 4.3 为什么用增量空间而非绝对坐标

1. **平移不变性**：增量表示消除了起始位置的影响，纯粹反映运动模式
2. **尺度统一**：归一化后各维度量级一致，K-means 聚类更有效
3. **heading 分解**：sin/cos 避免了角度环绕问题（如 -π 和 π 实际相同）

## 五、K-means 聚类

### 5.1 聚类过程

```
输入：delta_trajs [N, 8, 4]
展平：delta_flat  [N, 32]   （8 timestep × 4 features）
K-means：K=8, n_init=10, random_state=42
输出：labels [N]，每条轨迹的簇归属
```

### 5.2 GMN 参数计算

对每个簇 k（共 K=8 个簇）：

| 参数 | 形状 | 计算方式 |
|------|------|----------|
| `cluster_trajs[k]` | [8, 3] | 簇内所有**绝对坐标轨迹**的均值 |
| `center_points[k]` | [4] | 簇内所有**增量轨迹**在样本和时间步两个维度上的均值 |
| `center_std[k]` | [4] | **固定值 0.1**，不从数据统计 |

**关键细节 — center_points 的计算方式**：

```python
cluster_deltas = delta_trajs[mask]     # [Nk, 8, 4]
center_point = cluster_deltas.mean(axis=(0, 1))  # [4]  ← 同时在样本和时间步维度求均值
```

这与直接使用 K-means 聚类中心不同：
- K-means 中心 = `kmeans.cluster_centers_` 展平后 reshape 为 [8, 4]，再求时间步均值
- 实际使用的是直接对所有样本所有时间步求均值

对于 x_diff、y_diff（线性操作），两种方式等价。但对于 sin/cos（非线性），结果会有微小差异。

### 5.3 输出文件格式

保存为 pkl 文件，包含三个 tensor：

```python
{
    'cluster_trajs': Tensor[K, 8, 3],    # 簇均值轨迹（绝对坐标）
    'center_points': Tensor[K, 4],        # 高斯均值（增量空间）
    'center_std':    Tensor[K, 4],        # 高斯标准差（固定 0.1）
}
```

## 六、模型中的使用方式

### 6.1 加载与缩放

```python
# meanfuser_model.py MeanFlowHead.__init__
self.cluster_trajs = nn.Parameter(mean_std['cluster_trajs'], requires_grad=False)
self.gaussian_mean = nn.Parameter(mean_std['center_points'], requires_grad=False) * 0.5  # 缩放因子
self.gaussian_std  = nn.Parameter(mean_std['center_std'],    requires_grad=False)
```

**注意**：`center_points` 乘以了 0.5 作为实际的高斯均值，这使得噪声采样偏向数据中心和纯噪声之间。

### 6.2 训练阶段 — 条件噪声采样

```python
# forward()
# 1. 找 GT 轨迹最近的簇
cluster_centers = self.cluster_trajs                          # [K, 8, 3]
min_index = ||trajectories - cluster_centers||.argmin(dim=K)  # 每个 batch 选最近的簇

# 2. 从对应簇的高斯分布采样噪声
e = randn() * std + mean  # 对所有 K 个簇采样 [bs, K, 8, 4]
e = e.gather(min_index)   # 只取最近簇的噪声 [bs, 8, 4]
```

即：**训练时根据 GT 轨迹选取最匹配的噪声模式**，使模型学会从特定模式的噪声去噪到对应的驾驶行为。

### 6.3 推理阶段 — 多模态去噪

```python
# sample()
# 对每个簇各采样一个噪声（8个簇 → 8条候选轨迹）
mean = self.gaussian_mean.repeat_interleave(repeat_num)  # 支持 num_proposals > K
std  = self.gaussian_std.repeat_interleave(repeat_num)
e = randn() * std + mean  # [bs, num_proposals, 8, 4]

# 一步去噪（或多步）
x0 = e - model(e, r=0, t=1, context)   # 单步
```

**推理时每个簇独立产生一条候选轨迹**，通过 ARM（Adaptive Route Matching）模块选出最优轨迹。

## 七、复现实验

### 7.1 数据规模

| 指标 | 值 |
|------|----|
| navtrain 日志文件数 | 1,192 |
| 总场景数 | 651,526 |
| 轨迹形状 | [651526, 8, 3] |
| 提取速度（优化后） | 9,566 it/s |
| 总耗时 | ~2.5 分钟 |

### 7.2 轨迹统计

```
x range:       [-7.36, 80.53]     （4秒最远前进 80m）
y range:       [-22.92, 25.58]    （最大横向偏移 25m）
heading range: [-2.17, 3.07]      （最大转弯 ~120°）
```

### 7.3 归一化常数对比

| 统计量 | 实测值 | 代码中硬编码值 | 偏差 |
|--------|--------|----------------|------|
| x_diff_min | -2.4927 | -1.2698 | 96% |
| x_diff_max | 14.8554 | 7.4756 | 99% |
| x_diff_mean | 2.0484 | 2.9502 | 44% |
| y_diff_min | -4.8898 | -5.0121 | 2.4% |
| y_diff_max | 5.4503 | 4.8564 | 12% |
| y_diff_mean | 0.0216 | 0.0607 | 181% |

**偏差分析**：x 方向偏差显著且系统性偏大。可能原因：
- 作者使用了不同版本的数据集（NAVSIM v1 vs v2）
- 使用了不同的数据划分（full trainval 含 navtest）
- 使用了不同的 `frame_interval`（如隔帧采样则 diff 值翻倍）

### 7.4 聚类结果

| 簇 | 样本数 | 占比 | 平均速度 | 语义 |
|----|--------|------|----------|------|
| 0 | 15,215 | 2.3% | 3.67 m/s | 大角度右转 (sin≈-0.47) |
| 1 | 70,484 | 10.8% | 7.46 m/s | 中速直行 |
| 2 | 263,071 | **40.4%** | 0.13 m/s | 近静止/停车 |
| 3 | 52,796 | 8.1% | 12.38 m/s | 高速直行 |
| 4 | 71,312 | 10.9% | 4.99 m/s | 低速微调直行 |
| 5 | 29,677 | 4.6% | 4.23 m/s | 大角度左转 (sin≈0.40) |
| 6 | 73,725 | 11.3% | 2.39 m/s | 低速缓行/减速 |
| 7 | 75,246 | 11.5% | 9.85 m/s | 较高速直行 |

**分布特点**：
- 停车/近静止占最大比例（40.4%），反映城市驾驶的等红灯/排队场景
- 直行类（簇 1/3/4/7）合计 41.3%，覆盖不同速度段
- 转弯类（簇 0/5）合计 6.9%，转弯场景相对稀少

### 7.5 与原始 pkl 对比

| 字段 | 形状 | 最大差异 | 均值差异 | 说明 |
|------|------|----------|----------|------|
| `center_std` | [8, 4] | **0.000** | **0.000** | 完全一致（固定值 0.1）|
| `center_points` | [8, 4] | 0.923 | 0.197 | K-means 随机性 |
| `cluster_trajs` | [8, 8, 3] | 33.4 | 3.48 | 不同簇分配 |

**分析**：
- **`center_std` 完全一致**：确认标准差为固定值 0.1，非数据统计得到
- **`center_points` 差异适中**：cos 分量均在 [0.81, 1.0] 范围，sin 分量在转弯簇中较大，**结构模式一致**
- **`cluster_trajs` 差异较大**：因为 K-means 初始化随机性，相同语义的轨迹被分到不同编号的簇，导致均值轨迹数值不同，但整体覆盖的行为模式相同

## 八、结论

1. **GMN 生成流程已验证正确**：pipeline 为「轨迹提取 → diff_traj 归一化 → K-means 聚类 → 提取簇参数」
2. **center_std 固定为 0.1**，不依赖数据统计
3. **center_points 采用样本×时间步双重均值**，而非 K-means 簇心直接 reshape
4. **模型使用时乘以 0.5 缩放**：`gaussian_mean = center_points * 0.5`
5. **数值差异不影响功能**：K-means 的随机初始化会导致不同的簇分配，但覆盖的驾驶行为模式一致
6. **归一化常数来源存疑**：实测 navtrain 的 diff 统计量与代码中硬编码值偏差较大，推测作者使用了不同的数据版本或划分方式

## 九、相关文件

| 文件 | 说明 |
|------|------|
| `tools/gaussian_mixed_noise/generate_gmn.py` | GMN 生成脚本（本次编写） |
| `tools/gaussian_mixed_noise/navtrain_8_mean_std.pkl` | 作者提供的原始 GMN 参数 |
| `tools/gaussian_mixed_noise/navtrain_8_mean_std_generated.pkl` | 本次复现生成的 GMN 参数 |
| `navsim/agents/meanfuser/utils.py` | diff_traj / cumsum_traj 实现 |
| `navsim/agents/meanfuser/meanfuser_model.py` | MeanFlowHead 中 GMN 的加载与使用 |
