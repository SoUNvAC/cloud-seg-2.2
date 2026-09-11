# Cloud-Adapter 2.2：频率路由 Cross-Attention

本项目基于 TGRS 2025 Cloud-Adapter 源码，按 `02_frequency_routed_cross_attention.md` 完成实验 02 改造。DINOv2-L 与 Mask2Former 的外部输入输出保持不变；SPM context 被拆成 Spatial、Low-Frequency、High-Frequency 三路，再由逐 token、逐层 Router 融合三个等总 rank 的 cross-attention 专家。

## 本次改动

- 新增 `FrequencyDecomposer`：使用真实二维尺寸、固定 `cutoff_ratio=0.3`，FFT 在 AMP 下强制 FP32，支持矩形输入并缓存频率 mask。
- 新增共享 Q 的三专家注意力：主配置 rank 为 `8/4/4`，K/V/Out 独立低秩映射，总 rank 与原 rank 16 对齐。
- 新增 fixed、image-wise、token-wise 三种融合及负载均衡损失；每层独立 Router 和可训练 `alpha`，默认 `temperature=1.0`、`alpha=0.1`。
- cls token 不参与 FFT；单 Tensor 和四尺度 list/tuple cache 均受支持。
- adapter-only checkpoint 包含专家、Router、alpha、cutoff、temperature、routing mode、rank budget 和 balance 权重。
- 新增 Boundary F1（容差 3 pixels）与每 4000 iter 的 Router 监控文件 `router_stats.jsonl`。
- `configs/experiment_02/` 提供文档中的 A–G validation 消融，validation 使用 `val`，不会用 test 选模型。

## 建立环境

需要 Linux/WSL、Conda、支持 CUDA 11.8 的 NVIDIA 驱动。执行：

```bash
bash setup.sh
conda activate ca22
pytest -q tests
```

`setup.sh` 会创建或补全名为 `ca22` 的环境，安装 PyTorch 2.0.1 + CUDA 11.8、MMCV 2.1.0、MMSegmentation 1.2.2、MMDetection 3.3.0 以及测试和数据处理依赖。正式训练前必须确保全部测试通过；任何失败都不应启动 40k 训练。

## 数据与预训练权重

CloudSEN12_High_L1C 按以下结构放置：

```text
data/cloudsen12_high_l1c/
├── img_dir/{train,val,test}/*.png
└── ann_dir/{train,val,test}/*.png
```

将 DINOv2-L 转换权重放到：

```text
checkpoints/dinov2_converted_512x512.pth
```

如需转换原始权重：

```bash
python tools/convert_models/convert_dinov2.py INPUT.pth checkpoints/dinov2_converted_512x512.pth --height 512 --width 512
```

## 启动训练

先用 seed 42 完成 A–G validation 消融，例如主候选 E：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/experiment_02/e_token_balance_844.py
```

其他配置为 `a_original_rank16.py`、`b_static_844.py`、`c_image_router_844.py`、`d_token_router_844.py`、`f_token_balance_655.py`、`g_token_balance_466.py`；`e_token_balance_844_lambda_0_01.py` 用于 balance 权重 0.01，D 配置对应权重 0。按 validation mIoU 选择最佳 token-wise 配置后，用完全相同设置运行三个配对种子：

```bash
for SEED in 13 42 3407; do
  CUDA_VISIBLE_DEVICES=0 python tools/train.py CONFIG.py \
    --work-dir work_dirs/experiment_02/seed_${SEED} \
    --cfg-options randomness.seed=${SEED}
done
```

多 GPU 示例：

```bash
bash tools/dist_train.sh CONFIG.py 4 --cfg-options randomness.seed=42
```

训练后可生成 Router 图并在同一 GPU 上比较效率：

```bash
python tools/analyze_router_stats.py work_dirs/experiment_02/seed_*/router_stats.jsonl
python tools/benchmark_experiment_02.py \
  --reference-config configs/experiment_02/a_original_rank16.py \
  --reference-checkpoint BASELINE.pth \
  --frequency-config CONFIG.py \
  --frequency-checkpoint FR_STAR.pth
```

## 进入下一步的最低门槛

只有以下条件全部满足，实验 02 才值得进入后续步骤：

1. 三种子平均 mIoU 相对实际 reference baseline 至少 `+0.50`，且相对 static 1/3 至少 `+0.25`。
2. 至少 2/3 个配对种子提升，任何单种子退化不超过 `0.20 mIoU`。
3. Thin Cloud IoU 至少 `+1.00`，或四类 macro Boundary F1 至少 `+0.80`。
4. Clear Sky、Thick Cloud 任一类别平均 IoU 退化不超过 `0.40`。
5. adapter 参数不超过 `2.10M` 且不超过 DINOv2-L 的 `0.70%`；median latency 增幅不超过 `8%`，峰值显存增幅不超过 `10%`。
6. 超过 18/24 层不得出现专家权重 `>0.90` 或 `<0.02`；三个专家全局使用率均在 `[0.10, 0.70]`。
7. FFT 重建、矩形输入、Router、冻结主干、AMP、checkpoint 和 Boundary F1 测试全部通过。

精度达标但 Router 塌缩仍判定失败。详细实验约束和诊断见 `02_frequency_routed_cross_attention.md`，记录模板见 `EXPERIMENT_02_REPORT.md`。
