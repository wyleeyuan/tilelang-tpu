# paged_latent_attention_fp8 多核性能测试报告

日期：2026-07-01

## 结论摘要

本轮把 `paged_latent_attention_fp8` decode 路径里的 4 个 kernel 都做了 8 核并行验证，并额外减少了 attention kernel 中不必要的跨 core 同步。

当前完整 adapter E2E 通过，四个 kernel 的两次平均耗时约为：

| Kernel | 多核后耗时 1 | 多核后耗时 2 | 主要优化方式 |
| --- | ---: | ---: | --- |
| WUQ projection | 0.452 ms | 0.427 ms | 按输出 block/head 维度 8 核分片 |
| Absorb | 0.164 ms | 0.166 ms | 按 head 维度 8 核分片 |
| Attention | 0.266 ms | 0.273 ms | 按 head 维度 8 核分片，cache 更新只由 core0 执行 |
| WUV projection | 0.169 ms | 0.169 ms | 按 head 维度 8 核分片 |
| 合计 | 1.051 ms | 1.035 ms | 四 kernel 总耗时 |

相比最早只优化 attention/WUV 前的计时，前两个 kernel 从约 `2.6 ms + 0.7 ms` 降到约 `0.43-0.45 ms + 0.16 ms`；attention 从约 `1.60 ms` 降到约 `0.27 ms`；WUV 从约 `0.68 ms` 降到约 `0.17 ms`。整体从约 `5.5 ms` 级别降到约 `1.04-1.05 ms`，大约是 **5.2x-5.3x 总体加速**。

## 这个算子和 4 个 kernel

`paged_latent_attention_fp8` 是 decode 阶段的 paged latent attention 算子。它接收当前 token 的 latent query、KV/PE cache、page table、量化 scale 等输入，完成 query projection、cache 更新、paged cache gather、attention 计算和最终 value/output projection。

当前实现拆成 4 个 kernel：

| 顺序 | Kernel | 作用 |
| --- | --- | --- |
| 1 | `pla_wuq_projection_mc8_*` | 把 latent query 投影成每个 head 使用的 query 表示 |
| 2 | `pla_absorb_mc8_*` | 做 latent/weight absorb 相关中间计算 |
| 3 | `pla_attention_mc8_*` | 更新 KV/PE cache，从 paged cache gather 历史块，并计算 multi-head attention |
| 4 | `pla_wuv_mc8_*` | 把 latent attention 输出投影到 value/output head 维度 |

这 4 个 kernel 都有明显的 head 或输出块并行结构，所以本轮选择的主要方向是：保持原算子拆分不变，先把每个 kernel 内部的 head/block 循环切给最多 8 个 TPU core。

## 正确性验证

运行命令：

```bash
cd /mnt2/users/tilelanguser11/tilelang-tpu
source /mnt2/users/tilelanguser11/envs/tilelang_tpu_dev_v1.4.sh
PYTHONPATH=$PWD:$PWD/tpu_demo python3 /tmp/test_adapter_tpu_e2e_wuv_mc.py
```

最新验证结果：

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

说明多核分片没有破坏输出正确性，也没有破坏 KV/PE cache 的更新和第二次 decode 调用。

## 多核方案

### WUQ projection

WUQ projection 的输出维度可以按 block/head 切分。多核后 host launch 使用 8 个 core，生成 kernel 名称形如：

```text
pla_wuq_projection_mc8_h32_qr1536_kr512
```

它从最初约 `2.6 ms` 降到约 `0.43-0.45 ms`，大约 **5.8x-6.2x 加速**。

### Absorb

Absorb kernel 也按 head 分片，多核后名称形如：

```text
pla_absorb_mc8_h32_qr1536_kr512
```

它从约 `0.71-0.72 ms` 降到约 `0.164-0.166 ms`，大约 **4.3x 加速**。

### Attention

Attention kernel 是之前最明显的 head 串行瓶颈。原始逻辑近似为：

```c
for (int by = 0; by < 32; ++by) {
    ...
}
```

多核后改成最多 8 个 core，每个 core 负责 4 个 head。cache scatter/gather 只由 core0 做一次，其它 core 等待同步后读取 gather buffer。后续又去掉了输出阶段的额外同步，只保留 cache/gather 后必要同步。

它从约 `1.60 ms` 降到约 `0.266-0.273 ms`，大约 **5.9x-6.0x 加速**。

### WUV projection

WUV projection 按 head 独立计算，也适合直接做 8 核 head 分片。

它从约 `0.68 ms` 降到约 `0.169 ms`，大约 **4.0x 加速**。

## PPL 原版性能对比

我在服务器上找到的 PPL 原版相关目录是：

```text
/mnt2/users/tilelanguser11/ppl_v1.4.195-geb2acdd0-20250220/test_paged_latent_attention_ppl_v2
```

里面的原版 host wrapper 是单 kernel：

```text
paged_latent_attention_kernel
```

并且 host 侧写死：

```c
int core_num = 1;
int block_num = 1;
```

所以这个 PPL 原版测试本身不是多核版本。

### 能跑通的 PPL 测试

原版已有 artifact 是 `DEV_MODE=cmodel` 产物，需要用 bm1690 emulator runtime 跑：

```bash
cd /mnt2/users/tilelanguser11/ppl_v1.4.195-geb2acdd0-20250220/test_paged_latent_attention_ppl_v2
export PPL_PROJECT_ROOT=/mnt2/users/tilelanguser11/ppl_v1.4.195-geb2acdd0-20250220
export PPL_DATA_PATH=$PWD/data
export PPL_SRC_DIR_PATH=$PWD
export PPL_KERNEL_PATH=$PWD/lib/libcmodel.so
export LD_LIBRARY_PATH=$PPL_PROJECT_ROOT/runtime/bm1690/tpuv7-runtime-emulator_1.1.3/lib:$PPL_PROJECT_ROOT/runtime/bm1690/lib:$LD_LIBRARY_PATH
./test_case
```

我在临时副本 `/tmp/ppl_pla_perf_20260701` 中给 host test 加了 warmup + 100 次循环计时，加载原版 `libcmodel.so`，得到：

```text
PPL_ORIGINAL_TARGET_AVG_MS 741.859038
```

这个数字只能表示 **PPL 原版 cmodel/emulator 下的小 shape 单 kernel 调用时间**，不能和当前 TileLang PCIe 实机 kernel 时间做公平的绝对性能对比。

### 为什么不能直接公平对比

两边测试条件不同：

| 项目 | PPL 原版测试 | 当前 TileLang 测试 |
| --- | --- | --- |
| Runtime | cmodel/emulator | PCIe 实机 runtime |
| Kernel 形态 | 1 个 fused PPL kernel | 4 个 TileLang kernel |
| Core 数 | host wrapper 写死 `core_num = 1` | 4 个 kernel 都使用 8 core |
| 数据类型/shape | PPL 小 shape FP16 测试 | DeepSeek 适配路径 BF16/FP8 cache 形态 |
| 可比性 | 只能做结构和量级参考 | 当前真实优化目标 |

我也尝试过给 PPL 原版走 PCIe 构建，但当前目录没有可直接使用的 PCIe `libkernel.so`；用现有 CMake 配 `DEV_MODE=pcie` 会卡在 runtime/header/firmware 链接问题，不能得到可靠的 PPL PCIe 性能数字。

因此这次可信结论是：

1. PPL 原版测试已定位并运行成功，但可运行版本是 cmodel/emulator。
2. PPL 原版 host wrapper 确认是单 core launch，不具备当前多核优化。
3. 当前 TileLang 多核版本在目标 PCIe 实机路径下，四 kernel 总耗时约 `1.04-1.05 ms`。
4. 如果要做严格 PPL vs TileLang 性能对比，需要先拿到同 shape、同 dtype、同 runtime 的 PPL PCIe kernel artifact。

## 当前生成 artifact

关键多核 artifact 形如：

```text
src/tl_templates/tpu/kernel_pla_wuq_projection_mc8_h32_qr1536_kr512.*
src/tl_templates/tpu/main_pla_wuq_projection_mc8_h32_qr1536_kr512.*
src/tl_templates/tpu/libkernel_pla_wuq_projection_mc8_h32_qr1536_kr512.so

src/tl_templates/tpu/kernel_pla_absorb_mc8_h32_qr1536_kr512.*
src/tl_templates/tpu/main_pla_absorb_mc8_h32_qr1536_kr512.*
src/tl_templates/tpu/libkernel_pla_absorb_mc8_h32_qr1536_kr512.so

src/tl_templates/tpu/kernel_pla_attention_mc8_h32_qr1536_kr512.*
src/tl_templates/tpu/main_pla_attention_mc8_h32_qr1536_kr512.*
src/tl_templates/tpu/libkernel_pla_attention_mc8_h32_qr1536_kr512.so

src/tl_templates/tpu/kernel_pla_wuv_mc8_h32_qr1536_kr512.*
src/tl_templates/tpu/main_pla_wuv_mc8_h32_qr1536_kr512.*
src/tl_templates/tpu/libkernel_pla_wuv_mc8_h32_qr1536_kr512.so
```

## 后续建议

1. 如果继续追性能，优先看 WUQ projection。它已经多核，但仍是当前最大耗时，约 `0.43-0.45 ms`。
2. 如果要和 PPL 原版严格对比，需要先补齐 PPL PCIe 版 artifact，最好使用同一组 shape/dtype 输入。
3. 如果要长期维护当前多核 patch，建议把 head/block sharding 能力沉到 TileLang TPU codegen 或模板层，减少生成后字符串 patch 的维护成本。
