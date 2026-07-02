# paged_latent_attention_fp8 与 PPL 原版性能对比

日期：2026-07-02

## 结论

这次找到了更可信的 PPL 原版 PCIe 测试方式。昨天只能跑 cmodel/emulator，今天改成把 PPL 原版 `device/paged_latent_attention_ppl_v2.c` 重新编成 PCIe firmware `libkernel.so`，再写一个临时 C++ benchmark 按原版 host wrapper 的方式 launch。

测得：

| 项目 | Runtime | Kernel 形态 | 平均耗时 |
| --- | --- | --- | ---: |
| PPL 原版 `paged_latent_attention_ppl_v2` | PCIe runtime | 1 个 fused kernel，`core_num=1` | 0.593753 ms |
| 当前 TileLang `paged_latent_attention_fp8` | PCIe runtime | 4 个 kernel，均为 8 core 多核版本 | 1.034-1.063 ms |

这个结果不能直接理解为“谁快几倍”，因为两边不是同一个 shape/dtype/算子边界：

- PPL 原版是小 shape FP16 fused kernel。
- 当前 TileLang 是 DeepSeek 适配路径里的 BF16/FP8 cache 形态，包含 WUQ、Absorb、Attention、WUV 四段。
- PPL 原版 host wrapper 写死 `core_num=1`，当前 TileLang 四个 kernel 都是 8 core 多核。

更准确的说法是：**PPL 原版在它自己的小 shape 单核 fused kernel 上约 0.594 ms；当前 TileLang 在更大 DeepSeek 形态四 kernel 完整路径上约 1.05 ms。**

## Benchmark 分支检查

用户提到同学有一个 `Benchmark` 分支。服务器上 fetch 后没有直接叫 `Benchmark` 的分支，相关分支是：

```text
origin/performance-test
```

我用独立 worktree 检出检查：

```text
/tmp/tilelang_tpu_performance_test_wt
```

这个分支包含通用 TPU benchmark 框架：

```text
tpu_benchmark/
tpu_benchmark/_bench_worker.py
tpu_benchmark/ppl_utils.py
tpu_benchmark/performance_summary.md
```

其中 `ppl_utils.py` 的思路是把 PPL 生成的 kernel 包到 TileLang TPU host/runtime 路径里。它没有现成的 `paged_latent_attention_fp8` 对比脚本，但这个思路可复用，所以本次采用了类似方式：复用 PPL 原版 device C，手写一个最小 PCIe benchmark。

## PPL 原版测试对象

PPL 原版目录：

```text
/mnt2/users/tilelanguser11/ppl_v1.4.195-geb2acdd0-20250220/test_paged_latent_attention_ppl_v2
```

关键文件：

```text
device/paged_latent_attention_ppl_v2.c
host/paged_latent_attention_ppl_v2.cpp
include/paged_latent_attention_ppl_v2.h
```

原版 host wrapper 里是单核 launch：

```c
int core_num = 1;
int group_num = 1;
int block_num = 1;
tpuRtKernelLaunch(tpu_module, "paged_latent_attention_kernel", ...);
```

PPL 原版固定 shape：

| 输入/输出 | Shape | Dtype |
| --- | --- | --- |
| `q_latent` | `(1, 64, 1, 16)` | FP16 |
| `kv_latent` | `(1, 128, 1, 16)` | FP16 |
| `wuq` | `(1, 192, 1, 16)` | FP16 |
| `wukv` | `(1, 96, 1, 16)` | FP16 |
| `q_rope` | `(1, 64, 1, 16)` | FP16 |
| `k_rope` | `(1, 128, 1, 16)` | FP16 |
| `mask` | `(1, 16, 1, 64)` | FP16 |
| `out` | `(1, 64, 1, 32)` | FP16 |

## PPL PCIe benchmark 方法

临时目录：

```text
/tmp/ppl_pla_pcie_bench
```

生成文件：

```text
/tmp/ppl_pla_pcie_bench/libkernel.so
/tmp/ppl_pla_pcie_bench/bench_ppl_pla_pcie
```

运行命令核心形式：

```bash
PPL_TOP=/mnt2/users/tilelanguser11/ppl_v1.4.195-geb2acdd0-20250220
export LD_LIBRARY_PATH=/opt/tpuv7/tpuv7-current/lib:$PPL_TOP/runtime/bm1690/tpuv7-runtime-emulator/lib:$PPL_TOP/runtime/bm1690/lib:$LD_LIBRARY_PATH
/tmp/ppl_pla_pcie_bench/bench_ppl_pla_pcie /tmp/ppl_pla_pcie_bench/libkernel.so 14 5 20
```

其中：

- device id: `14`
- warmup: `5`
- measure runs: `20`
- 计时范围：单次 `tpuRtKernelLaunch(...)` + `tpuRtStreamSynchronize(stream)`
- 不包含 host 随机初始化、S2D/D2S 拷贝和 device malloc/free

输出：

```text
PPL_PCIE_ORIGINAL_SHAPE q=64 kv=128 wuq=192 wukv=96 out=64x32
PPL_PCIE_ORIGINAL_CORE_NUM 1
PPL_PCIE_ORIGINAL_RUNS warmup=5 runs=20
PPL_PCIE_ORIGINAL_AVG_MS 0.593753
PPL_PCIE_ORIGINAL_MIN_MS 0.580420
PPL_PCIE_ORIGINAL_MAX_MS 0.623588
```

## 当前 TileLang 多核性能

运行命令：

```bash
cd /mnt2/users/tilelanguser11/tilelang-tpu
source /mnt2/users/tilelanguser11/envs/tilelang_tpu_dev_v1.4.sh
PYTHONPATH=$PWD:$PWD/tpu_demo python3 /tmp/test_adapter_tpu_e2e_wuv_mc.py
```

正确性：

```text
model adapter output max diff: 0.000031
output finite: True
output allclose: True
KV cache update: True
PE cache update: True
second call finite: True
second call stable: True
Soph DeepSeek V3 TileLang model-call adapter: True
```

最新两轮四 kernel 性能：

| Kernel | 第 1 轮 | 第 2 轮 |
| --- | ---: | ---: |
| WUQ projection | 0.439 ms | 0.431 ms |
| Absorb | 0.185 ms | 0.168 ms |
| Attention | 0.266 ms | 0.259 ms |
| WUV projection | 0.173 ms | 0.176 ms |
| 合计 | 1.063 ms | 1.034 ms |

## TileLang 四 kernel 单核 vs 多核

为了回答“四个 kernel 原版和现在多核版本差多少”，我又生成了一个临时单核模块：

```text
/tmp/paged_latent_attention_fp8_single.py
/tmp/test_adapter_tpu_e2e_single.py
```

这个临时模块只去掉 `mc8` 多核 patch，shape、dtype、四 kernel 拆分和 E2E 输入都保持与当前 TileLang 版本一致。因此它可以作为四 kernel 的公平单核基线。

单核正确性同样通过：

```text
model adapter output max diff: 0.000031
output finite: True
output allclose: True
KV cache update: True
PE cache update: True
second call finite: True
second call stable: True
Soph DeepSeek V3 TileLang model-call adapter: True
```

逐 kernel 对比：

| Kernel | 单核平均 | 多核平均 | 加速比 |
| --- | ---: | ---: | ---: |
| WUQ projection | 2.636 ms | 0.435 ms | 6.06x |
| Absorb | 0.702 ms | 0.177 ms | 3.98x |
| Attention | 1.596 ms | 0.263 ms | 6.08x |
| WUV projection | 0.681 ms | 0.175 ms | 3.90x |
| 合计 | 5.615 ms | 1.049 ms | 5.35x |

这里的“单核平均”和“多核平均”都是两轮 E2E benchmark 的平均值。多核版本整体耗时下降约 `81%`。

## 怎么解读

如果只看绝对数字：

```text
PPL 原版小 shape fused kernel: 0.594 ms
TileLang 当前 DeepSeek 四 kernel 单核: 5.60-5.63 ms
TileLang 当前 DeepSeek 四 kernel 多核: 1.03-1.06 ms
```

但这个绝对对比不是公平同 shape benchmark。PPL 原版做的是更小的 FP16 固定 shape fused kernel；TileLang 当前路径做的是更大维度的 DeepSeek decode 适配，并且拆成 4 个 kernel。它们更适合回答两个不同问题：

1. PPL 原版在自己的测试形态下有多快？
   - 答案：PCIe runtime 下约 `0.594 ms`。

2. 当前 TileLang 多核版本在目标模型适配路径下有多快？
   - 答案：四 kernel 总计约 `1.05 ms`，正确性通过。

3. 当前四 kernel 从单核到多核提升多少？
   - 答案：从约 `5.615 ms` 到约 `1.049 ms`，整体约 `5.35x`。

4. 和 PPL 原版的差距如何看？
   - 按绝对值看，当前 TileLang 多核四 kernel 约为 PPL 原版小 shape fused kernel 的 `1.77x` 耗时。
   - 但这不是严格劣化结论，因为 TileLang 的 shape 更大、dtype/量化路径不同，并且包含 4 次 kernel launch。

如果后续要做严格的一对一性能结论，需要补一个同 shape、同 dtype、同输入输出语义的 TileLang 版本，或者把 PPL 原版扩到当前 DeepSeek 的 BF16/FP8 cache 形态。
