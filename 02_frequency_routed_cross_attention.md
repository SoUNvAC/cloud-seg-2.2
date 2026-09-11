# 实验 02：空间/低频/高频路由式 Cross-Attention

## 1. 目标与硬性结论

把 Cloud-Adapter 的单一 SPM context 改造成 Spatial、Low-Frequency、High-Frequency 三路 context，并通过逐 token、逐层 Router 动态融合三个 cross-attention 专家。

核心假设：

- LF 分支更适合大片云体、薄云雾化和云影的全局结构；
- HF 分支保留云边界和细节，但也包含地表纹理与传感器噪声；
- Spatial 分支保留未经过频域截断的完整信息；
- 动态 Router 应优于固定等权融合，且不能塌缩到单一专家。

只有第 12 节全部条件满足才算实验成功。

## 2. 输入基线

优先使用实验 01 成功模型 `LS*`，记其三种子均值为 `B1`。

如果实验 01 失败，可以从原始复现基线 `B0` 开始，但必须在报告中写明：

```text
Experiment-01 dependency: FAILED
Current reference baseline: B0
```

后续所有增量均相对实际采用的 reference baseline 计算。

固定设置：CloudSEN12_High_L1C、DINOv2-L、Mask2Former、512 输入、40k iter、batch 4、三个配对种子 `13/42/3407`。

## 3. 推荐架构

设当前 Transformer token 为 `F_l in R^(B x N x D)`，SPM 当前尺度 context 为 `C_l in R^(B x C x H x W)`。

### 3.1 动态二维频域分解

必须从真实 tensor shape 获得 `H,W`，禁止写死 `32 x 32`：

```python
fft = torch.fft.fft2(context.float(), dim=(-2, -1), norm="ortho")
fft = torch.fft.fftshift(fft, dim=(-2, -1))
mask_low = build_mask(H, W, cutoff_ratio=0.3, device=context.device)
low = torch.fft.ifft2(
    torch.fft.ifftshift(fft * mask_low, dim=(-2, -1)),
    dim=(-2, -1), norm="ortho"
).real.to(context.dtype)
high = context - low
spatial = context
```

本实验固定使用 `cutoff_ratio=0.3`。可学习 cutoff 留到实验 04，禁止在本实验混入。

FFT 在 AMP 下统一转为 FP32 计算，再转回原 dtype，避免半精度复数不稳定。

### 3.2 三路 Cross-Attention 专家

```text
Delta_sp = CA_sp(Q=F_l, K=C_sp, V=C_sp)
Delta_lf = CA_lf(Q=F_l, K=C_lf, V=C_lf)
Delta_hf = CA_hf(Q=F_l, K=C_hf, V=C_hf)
```

为控制参数量：

- 三个专家共享 `Q projection`；
- K、V、Out 使用独立低秩映射；
- 主配置的 rank budget 为 `sp/lf/hf = 8/4/4`，三者之和等于原 Cloud-Adapter 的 rank 16；
- 另做 `6/5/5` 和 `4/6/6` 单种子 validation 消融；
- 禁止三个专家各自使用完整 rank 16 作为主结果。

建议新增：

```text
FrequencyDecomposer
SharedQueryMultiExpertCrossAttention
TokenFrequencyRouter
FrequencyRoutedInteractiveModule
```

### 3.3 Token-wise Router

Router 输入同时包含当前 VFM token 和 SPM 全局摘要：

```python
ctx_global = context.mean(dim=(-2, -1))       # B,C
ctx_global = self.ctx_proj(ctx_global)        # B,router_dim
ctx_global = ctx_global[:, None, :].expand(-1, N, -1)
router_in = torch.cat([self.token_norm(x), ctx_global], dim=-1)
logits = self.router(router_in)               # B,N,3
weights = torch.softmax(logits / temperature, dim=-1)
delta = sum(weights[..., k:k+1] * delta_k for k in range(3))
out = x + alpha_l * delta
```

- `temperature=1.0` 为主设置。
- `alpha_l` 沿用实验 01；若实验 01 失败，初始化为 0.1。
- Router 为每层独立参数，不共享。
- 不允许只做整图三个权重后声称“token-wise routing”。

### 3.4 防塌缩损失

在 segmentation loss 外增加轻量负载均衡项：

```text
p_k = mean(weights over batch and tokens)
L_balance = sum_k (p_k - 1/3)^2
L_total = L_seg + lambda_balance * L_balance
```

单种子 validation 比较 `lambda_balance in {0, 0.001, 0.01}`，主候选不得使用 test 选择。

## 4. 代码改动位置

```text
cloud_adapter/models/backbones/cloud_adapter.py
cloud_adapter/models/backbones/cloud_adapter_dinov2.py
configs/experiment_02/*.py
```

要求：

- `CloudAdapter.forward` 的外部输入输出 shape 不变；
- 仍支持 cache 是单 Tensor 或四尺度 list/tuple；
- 保留 cls token，不对 cls token 做二维 FFT；
- 非正方形输入必须通过真实 H、W 运行，不得使用 `sqrt(N)` 猜测；
- FFT mask 注册为 buffer 或按 shape/device 缓存，不得每层进行 CPU 构造和传输；
- 保存 adapter-only checkpoint 时包含专家、Router、alpha 和频率配置。

## 5. 单元测试

训练前必须通过：

1. `low + high` 与原 context 的最大绝对误差 `< 1e-5`（FP32）。
2. 输入 shape `(2,64,32,32)`、`(2,64,16,16)`、`(1,64,24,32)` 均可前反向传播。
3. Router 权重最后一维之和与 1 的误差 `< 1e-6`。
4. 冻结 DINOv2 参数无梯度。
5. AMP 前向无 NaN/Inf。
6. adapter-only checkpoint 保存后重载，同一输入输出最大误差 `< 1e-5`。

任一单元测试失败，不得开始正式训练。

## 6. 必跑消融

先用 seed=42 在 validation 上执行：

| ID | 频率分支 | 融合方式 | Router 粒度 | Rank |
|---|---|---|---|---|
| A | 无 | 原 Cloud-Adapter | 无 | 16 |
| B | SP/LF/HF | 固定 1/3 | 无 | 8/4/4 |
| C | SP/LF/HF | 动态 | image-wise | 8/4/4 |
| D | SP/LF/HF | 动态 | token-wise | 8/4/4 |
| E | SP/LF/HF | 动态+balance | token-wise | 8/4/4 |
| F | SP/LF/HF | 动态+balance | token-wise | 6/5/5 |
| G | SP/LF/HF | 动态+balance | token-wise | 4/6/6 |

按 validation mIoU 选择最佳 token-wise 模型 `FR*`，然后运行三随机种子 test。B、最佳 image-wise、最佳 token-wise 都必须保留结果，才能验证 Router 的贡献。

## 7. 数据与训练设置

- 数据、增强、optimizer、scheduler、decoder 和训练步数与 reference baseline 完全相同。
- 允许新增 `L_balance`，除此之外不增加任何辅助监督。
- 所有模型从相同 DINOv2 checkpoint 初始化。
- 每个配对 seed 使用完全相同的数据顺序；必要时保存 sampler index。
- validation 每 4,000 iter；主 checkpoint 只由 validation mIoU 选择。

## 8. 边界指标

除官方指标外，计算 Boundary F1：

1. 对 GT 和预测 mask 分别提取形态学边界；
2. 容差宽度固定为 3 pixels；
3. 四类分别计算，再取 macro average；
4. 同时单独报告 Thin Cloud 和 Cloud Shadow 的 Boundary F1。

指标实现必须先在完全相同 mask 上自测得到 1.0。

## 9. Router 监控

每个 epoch/4,000 iter 保存：

- 每层三个专家的平均使用率；
- 每层归一化 Router entropy；
- 每类像素对应的平均专家权重；
- token 间权重方差；
- 三路 `delta` 的 L2 norm。

Router 塌缩定义：训练结束时，在超过 18/24 层中，任一专家平均权重超过 0.90，或任一专家平均权重低于 0.02。

## 10. 效率测试

对 reference baseline 和 `FR*` 使用相同环境：

- 统计 adapter 部分 trainable params；
- 统计整个模型 forward FLOPs；
- batch=1，512 输入，FP16，100 次预热，500 次计时；
- 报告 median/P90 latency；
- 报告 peak allocated GPU memory。

计时期间关闭可视化、Router 日志和 checkpoint hook。

## 11. 必须提交的表和图

- A-G validation 消融表；
- 三种子 test 配对表；
- 各类别 IoU 和 Boundary F1；
- 24 层专家平均权重热图；
- Thin Cloud、Thick Cloud、Cloud Shadow、Clear Sky 四类的专家权重箱线图；
- 参数量、显存、median/P90 latency 表。

## 12. 硬性成功标准

以下全部满足才成功：

1. `mean(mIoU_FR*) >= reference + 0.50`。
2. `mean(mIoU_FR*) >= mean(mIoU_static_1/3) + 0.25`。
3. 至少 2/3 个配对种子提升，任一种子退化不超过 `0.20 mIoU`。
4. Thin Cloud IoU 至少提高 `1.00`，或四类 macro Boundary F1 至少提高 `0.80`；两者至少满足一个。
5. Clear Sky 和 Thick Cloud 的任一类别平均 IoU 退化不得超过 `0.40`。
6. adapter trainable params 不超过 `2.10M`，且不超过冻结 DINOv2-L 参数量的 `0.70%`；统计口径须与原论文 1.82M 一致。
7. median latency 增幅不超过 `8%`，峰值显存增幅不超过 `10%`。
8. Router 不得达到第 9 节的塌缩定义；三个专家的全局平均使用率均须位于 `[0.10, 0.70]`。
9. FFT 重建、AMP 和 checkpoint 单元测试全部通过。

若精度过线但 Router 塌缩，本实验仍判失败，因为不能支持“动态多专家”机制主张。

## 13. 失败诊断

- B 比 A 低：频率分解或等参数分配本身无效，先检查重建误差、mask 中心和 context shape。
- B 有效、D/E 无效：Router 设计失败；检查温度、归一化与 context 摘要。
- Router 塌缩：仅允许在新 run 中调整 `lambda_balance` 或 temperature，不得覆盖失败 run。
- Thin Cloud 提高但 Clear Sky 大幅下降：高频背景纹理被误判；检查 HF 专家和 Router 是否在非云区过度激活。
- 参数或速度超线：优先共享 Q/K 投影或缓存 FFT；禁止直接放宽验收标准。

## 14. 交付物

```text
cloud_adapter/models/backbones/frequency_routed_adapter.py
configs/experiment_02/*.py
tests/test_frequency_decomposer.py
tests/test_frequency_router.py
work_dirs/experiment_02/summary.csv
work_dirs/experiment_02/router_layer_heatmap.png
work_dirs/experiment_02/router_class_boxplot.png
work_dirs/experiment_02/efficiency.json
EXPERIMENT_02_REPORT.md
```

