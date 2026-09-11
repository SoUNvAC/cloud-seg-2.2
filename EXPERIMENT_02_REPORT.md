# Experiment 02 Report

Status: **NOT RUN**

```text
Experiment-01 dependency: NOT REPORTED
Current reference baseline: NOT SET (B1 if experiment 01 passed; otherwise B0)
```

不得使用 test 选择超参数。先完成 seed 42 的 A–G validation，再对选出的 `FR*` 和 reference baseline 运行配对种子 `13/42/3407`。

## 训练前门禁

- [ ] `pytest -q tests` 全部通过
- [ ] CloudSEN12_High_L1C 的 train/val/test 划分存在
- [ ] 所有 run 使用相同 DINOv2-L 初始权重、数据增强、采样顺序、optimizer、scheduler、decoder、batch 4、512 输入和 40k iter

## Validation 消融

| ID | 配置 | seed | val mIoU | Thin Cloud IoU | macro Boundary F1 | 结论 |
|---|---|---:|---:|---:|---:|---|
| A | original rank16 | 42 |  |  |  |  |
| B | static 1/3, 8/4/4 | 42 |  |  |  |  |
| C | image router, 8/4/4 | 42 |  |  |  |  |
| D | token router, 8/4/4 | 42 |  |  |  |  |
| E | token + balance, 8/4/4 | 42 |  |  |  |  |
| F | token + balance, 6/5/5 | 42 |  |  |  |  |
| G | token + balance, 4/6/6 | 42 |  |  |  |  |

Selected FR*: **NOT SET**

## 三种子 test 与效率

将逐 seed、逐类别 IoU、Boundary F1 填入 `work_dirs/experiment_02/summary.csv`。效率结果填入 `work_dirs/experiment_02/efficiency.json`，并保留相同硬件上 reference 与 FR* 的 adapter 参数量、FLOPs、FP16 batch-1 median/P90 latency 和 peak allocated memory。

Router 图由每个 run 的 `router_stats.jsonl` 生成，目标文件为：

- `work_dirs/experiment_02/router_layer_heatmap.png`
- `work_dirs/experiment_02/router_class_boxplot.png`

## 是否进入下一步

结论：**PENDING**。只有 README 所列七项硬性标准全部通过才改为 **PASS**；任一项未通过均为 **FAIL**，不得用 test 重新选择 Router、temperature 或 balance 权重。
