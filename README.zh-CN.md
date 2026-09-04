<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center"><a href="README.md">English</a> | <b>简体中文</b></p>

# Nano-vLLM — 我的扩展

一个轻量级 vLLM 风格推理引擎,**fork 自
[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)**(MIT,© Xingkai Yu)。
上游快照(约 1,200 行:continuous batching、分页 KV、CUDA graph、TP、前缀缓存)是本仓库的
`Initial commit`;**其后全部为本人的工作** —— 新增约 4,900 行,实现了混合线性注意力模型、
多模态推理、流水线/数据并行、vLLM-V1 风格异步调度、投机解码以及一组内核级优化,
每一项都对照参考实现做了验证。

## 我在上游之上做了什么

| 领域 | 内容 | 代码位置 | 验证方式 |
|---|---|---|---|
| **混合模型(Qwen3.5-2B)** | 移植 gated-delta-net + 稀疏注意力混合架构:循环状态池化、因果卷积前缀、带输出门的 partial RoPE | `nanovllm/models/qwen3_5.py` | `tests/ref_check_qwen35.py` —— prefill logits + greedy 解码对照 transformers |
| **多模态(Qwen3.5 视觉)** | 完整 `Qwen3_5ForConditionalGeneration`:视觉塔、embedding scatter、交错式 MRoPE、跨 chunk 安全的图像暂存 | `qwen3_5_vision.py`、`utils/multimodal.py`、`MRotaryEmbedding` | `tests/mm_vision_parity_qwen35.py`(29 个阶段**位级一致**)、`tests/mm_check_qwen35.py`(分块 prefill / 混合批 / TP / PP / async 下与 HF 逐 token 一致) |
| **GDN 性能** | varlen prefill(fla chunk 内核 + varlen 因果卷积 Triton 内核)、融合 g/β 内核、原地循环解码(免去 450MB 状态 gather/scatter)、FlashInfer 融合归一化 | `qwen3_5.py` | 位级等价 A/B 开关(`NANOVLLM_GDN_*`);单层 prefill 8–30×,decode 约 1.4× |
| **异步调度(vLLM V1 风格)** | GPU token 环、滞后输出处理、输出占位符式乐观调度 | `engine/async_scheduler.py`、`model_runner.py` | `tests/async_check.py`;decode 吞吐 +6~59% |
| **并行** | 在上游 TP 之上实现 PP(按层切分、融合残差传递)与 DP(引擎复制、LPT 装箱),含混合模型状态切分 | `utils/parallel.py`、`engine/*` | `tests/parallel_check.py` —— tp/pp/dp 各布局 prefill 逐 token 一致 |
| **投机解码** | Jacobi 并行草稿 + 经典草稿模型 + DFlash 块扩散草稿,严格逐位置拒绝采样验证 | `model_runner.py`、`models/dflash.py`、`benchmarks/verify_spec.py` | 分布单元测试;负面结果如实记录(见下) |
| **Bug 猎捕** | 借多模态测试发现并修复两个混合引擎隐性正确性 bug(详见下文) | `block_manager.py`、`model_runner.py` | 修复前均可稳定复现,修复后全绿 |

## 快速开始

```bash
# attention 内核:flash-attn wheel,或 vLLM 内置的 FA2 内核
# (部分 torch/CUDA 组合没有 flash-attn wheel,如 cu13 + sm120)
pip install -e ".[flash]"            # 你的环境有 flash-attn wheel 时
pip install -e ".[vllm-fallback]"    # 否则:安装 vllm,使用 vllm.vllm_flash_attn

huggingface-cli download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
python example.py                    # 或:pip install git+https://github.com/sileaver/nano-vllm-extend.git
```

API 与 vLLM 对齐(`LLM.generate` 返回 token id + 文本):

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(["Hello, Nano-vLLM."],
                       SamplingParams(temperature=0.6, max_tokens=256))
outputs[0]["text"]
```

示例/测试中的模型路径默认 `~/huggingface/...`;Qwen3.5 相关脚本可用
`QWEN35_MODEL=/path` 覆盖。

## 多模态(Qwen3.5 视觉)

Qwen3.5 检查点以多模态外壳发布;引擎完整加载 —— 视觉塔、MRoPE 一应俱全。图像 prompt
采用与 vLLM 相同的 dict 形式(每张图在文本里对应一个
`<|vision_start|><|image_pad|><|vision_end|>` 占位符,由检查点自带的 processor
按图像网格展开为对应数量的 token):

```python
prompts = [
    {"prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
               "Describe this image.<|im_end|>\n<|im_start|>assistant\n",
     "images": ["path/or/PIL.Image"]},
    "同一引擎也接受纯文本 prompt",
]
outputs = llm.generate(prompts, sampling_params)   # example_qwen35_mm.py
```

底层实现:

* **视觉塔**(Qwen3-VL 风格 ViT:Conv3d patch embed、按图网格双线性重采样的学习位置表、
  逐帧非因果 flash attention + 2D rope、2×2 patch merger)移植于
  `nanovllm/models/qwen3_5_vision.py` —— fp32 下对 transformers 参考实现的
  **全部 29 个阶段位级一致**。视觉塔在 TP 各 rank 复制、仅存在于流水线第一 stage;
  融合后的 patch embedding 替换(经 all-reduce 的)token embedding 中的
  `image_token_id` 行 —— 每个序列只做一次视觉塔前向,无论 prefill 被切成多少 chunk。
* **交错式 MRoPE**(`MRotaryEmbedding`)—— prefill 在图像区域携带 [3, N] 的
  T/H/W 位置表(`get_rope_index` 的移植位于 `utils/multimodal.py`);decode 通过
  每序列的 `rope_delta` 折回 1D,CUDA graph 解码回放完全不受影响。
* 像素以 bf16 传输(无损 —— 参考实现本来就会转换到塔的 dtype),且只随 prefill
  状态序列化;TP shm 命令段已相应扩容。
* `NANOVLLM_QWEN35_TEXTONLY=1` 跳过视觉塔(即上游行为)。

### 多模态测试挖出的两个隐性 bug

两者都是*任何*混合(线性注意力)引擎的固有陷阱,此前在纯文本路径就存在,
只有跨实现的一致性对拍才能暴露:

1. **前缀缓存复用破坏混合状态。** 哈希命中的 KV 块让重复 prompt 从序列中段恢复
   prefill,但 GDN 层的循环状态无法从缓存 KV 重建 —— 续写会静默跑偏。修复:混合
   模型禁用哈希前缀复用(`BlockManager(enable_prefix_cache=False)`),与 vLLM
   对 mamba 类层的选择一致。
2. **CUDA graph 填充行污染真实槽位。** 以 `bs <` 捕获尺寸回放图时,残留行带着
   捕获期的 `linear_state_ids = 0` —— 一个*真实*槽位 —— 原地循环内核每次回放
   都用脏数据破坏该序列的状态(只有 slot-0 的序列跑偏,且只在开图时)。修复:
   增加专用 dummy 槽位,填充/捕获行统一指向它(分页 KV 写入早有类似的 `-1` 守卫)。

## 并行(TP / PP / DP)

三种策略可自由组合(`dp * pp * tp` 个 rank,每 rank 一卡,单机)。在 2× RTX 5090
上对照单卡验证:各布局 prefill argmax 逐 token 一致(32/32 真实 prompt;见
`tests/parallel_check.py`、`tests/real_prompt_check.py`):

```python
llm = LLM(path, tensor_parallel_size=2)    # TP:每 rank 切分 head/神经元
llm = LLM(path, pipeline_parallel_size=2)  # PP:按层切分到各 stage
llm = LLM(path, data_parallel_size=2)      # DP:引擎复制,各自带调度器
```

* **TP** 切分 attention/MLP 投影、词表 embedding、GDN 头(卷积通道 + 按 v-head
  粒度的循环状态)与 KV cache;每层一次 all-reduce。
* **PP** 按层切分到各 stage(stage 间 `dist.send/recv` 传递融合的 hidden+residual
  链)。每个 stage 只持有自己那部分层的 KV/循环状态池 —— pp=2 时单卡显存约减半;
  block/slot 数量跨 stage 取 min 同步;decode CUDA graph 按 stage 捕获。v1 同步
  执行单批(无 micro-batch 重叠)—— 用于容量,不用于延迟。
* **DP** 每 GPU 组复制完整引擎;driver 端 LPT 装箱分发请求并按序归并结果 ——
  512 个长短不齐的请求在 2× RTX 5090 上 1.94×(256 个请求时仅 1.38×,副本
  处于 batch 饥饿状态)。

投机解码要求 `tp = pp = 1`(可在 DP 下运行)。逐步 `add_request`/`step`
流式接口与 `collect_timing` 仅支持单组。

## 异步调度(vLLM V1 风格)

`async_scheduling=True` 让 CPU 调度与 GPU 执行流水化 —— 采样 token 不再往返 CPU:

* **GPU token 环** —— 采样 id 留在 GPU;下一个 decode 步在设备上从上一步的槽位
  gather 输入 id,跨 TP/PP 通过一次 NCCL broadcast 移动。
* **滞后输出处理** —— 结果经异步 D2H 写入由 CUDA event 守护的 pinned memory;
  引擎滞后 1–2 步应用,完全移出关键路径(流水深度上限 2)。
* **乐观调度** —— 在飞 token 的 KV 槽位用输出占位符预留(vLLM 的 "future token
  ids");命中 EOS 的序列会在 CPU 察觉前多跑一行。
* **统一流水线** —— prefill 块同样预留占位符,因此走同一条不排空的流水:chunk
  续算、decode 叠加未收割 prefill、混合批输入组装,全部由同一个
  `num_computed_tokens` 不变式推出。

RTX 5090 实测(decode 密集):Qwen3-0.6B @ bs=64 20.1k → 22.6k tok/s(+11.6%),
Qwen3.5-2B @ bs=64 10.2k → 11.1k tok/s;并行度越高收益越大(有更多 CPU/NCCL
开销可隐藏):Qwen3-0.6B @ tp2 11.9k → 19.0k(+59%),Qwen3.5-2B @ tp2
10.1k → 11.4k(+11.7%)。

配合下文的 GDN 优化,在长短不齐的 serving 负载上达到与 vLLM 0.27 持平 —— 混合
吞吐在这轮重构中从 **9.5k → 22.7k tok/s(约为 vLLM 的 41% → 98%)**(完整对比
表见下文 Benchmark 一节)。

## 混合模型(GDN)优化

* **Varlen GDN** —— 该层原来把 batch pad 成 `[bs, max_query_len]`,在 decode+prefill
  混合步上最多浪费约 150× 计算。decode 行走 O(1) 循环路径;prefill 组全 varlen
  (稠密 `[N, H]` 投影、varlen 因果卷积 Triton 内核、由 `cu_seqlens` 驱动的 fla
  chunk 内核 —— 已验证与 padded 调用位级等价)。
* **融合管线** —— 融合 g/β Triton 内核取代 eager fp32 逐元素链;Gemma 风格归一化
  与 MLP 激活跑在 FlashInfer 融合内核上;decode CUDA graph 一并捕获 lm_head +
  两段式融合 gumbel-max 采样内核。
* **原地循环解码** —— decode 步原来要 gather 450MB 状态批、跑循环、再 scatter
  回去;改为按池行号原地更新的专用内核(bs≈218 时省下约 40% decode 时间)。
* **零拷贝 GDN prefill 管线** —— 对照 vLLM 做 profile 发现 fla chunk 内核的输入
  守卫在重复拷贝 q/k/v(每层 3×64MB,约占 prefill 时间 1.7%):varlen 因果卷积
  内核现在把输出写为三块各自连续的 `[N, H, D]`,下游零拷贝。RoPE 与 attention
  输出门同样处理 —— 各用一个原地 Triton 内核取代约 16 个内核的 eager 链加整头
  拷贝(与 eager 形式位级一致;`NANOVLLM_FUSED_ROPE=0` 可 A/B)。净效果:prefill
  61.0k → 63.0k tok/s,混合模型从 −1.8% 变为 **+1.5%**(对 vLLM)。
* **单 token 卷积内核** —— decode 步原来用 eager 的 gather → cat → conv1d →
  scatter 链滚动因果卷积状态(四次全状态显存往返,每个混合负载各约 1.8 万次 cat
  和 scatter 内核 —— 只在混合 serving 里显现:decode 行与 prefill 块同批时落在
  CUDA graph 之外)。专用 Triton 内核一次完成窗口卷积 + SiLU + 状态滚动(状态
  滚动位级一致,bs≈218 时每层 2.6×):mixed 22.8k → 24.1k tok/s(**vLLM 的
  104%**),纯 decode 20.4k → 21.9k(**109%**)。

## 投机解码

两种模式,由 `num_spec_tokens=K` 开启,严格逐位置拒绝采样验证(被接受序列的分布
可证明与普通自回归采样一致;`benchmarks/verify_spec.py`):

```python
llm = LLM(path, num_spec_tokens=4)                      # Jacobi 并行草稿
llm = LLM(target, num_spec_tokens=4, spec_draft_model=draft)  # 草稿模型
```

如实记录的负面结果(记录,不隐藏):

* **0.6B–4B 规模无收益**:Jacobi 草稿比 CUDA-graph 基线慢 2–4×;0.6B→4B 草稿
  投机的接受率只有 0.05–0.13。每步要付两次多行前向 + 151k 词表 softmax ——
  要 7B+ 的目标模型才可能回本。
* **跳层自投机失败**:仅跳一层就把草稿匹配率从 ~0.5 打到 ~0.14 —— 这两个模型
  没有层冗余。
* **DFlash 块扩散草稿已移植集成**(`models/dflash.py`)—— 相同输入下与官方实现
  位级一致,但同样 prompt 上接受率比官方低 10–13×:草稿以*目标模型*的隐状态为
  上下文,bf16 重实现的舍入差异(在最终 logits 中不可见)经其注意力逐层放大。
  已作为非位级一致重实现的固有限制记录,并逐层测量了放大链。

## 项目结构

```
nanovllm/
├── engine/            # 调度器(legacy/continuous/async)、model runner、
│                      #   block manager、shm 广播 worker 循环
├── layers/            # attention(分页 KV + flash varlen)、rope/MRoPE、
│                      #   并行 linear、sampler(+ gumbel triton 内核)
├── models/            # qwen3(稠密)、qwen3_5(混合 GDN + attention)、
│                      #   qwen3_5_vision(ViT 塔体)、dflash(草稿)
└── utils/             # loader、并行状态、多模态预处理
benchmarks/            # 吞吐对比(nano vs vLLM)、投机解码验证
tests/                 # 对照 transformers / 跨并行布局 / 异步的一致性检查;
                      #   ref_mm/ 重新生成多模态黄金基准
example*.py            # 文本、混合模型与多模态用法示例
```

## 验证

每个子系统都有可运行的参考实现对照检查(2× RTX 5090、torch 2.13 / cu130 全绿):

```bash
python tests/ref_check_qwen35.py          # 混合 LM:prefill logits + greedy 解码对照 HF
python tests/ref_mm/gen_reference.py      # 重新生成多模态黄金基准(约 310MB,已 gitignore)
python tests/mm_vision_parity_qwen35.py   # 视觉塔:29 个阶段对照 HF 位级一致(fp32)
python tests/mm_check_qwen35.py           # 多模态端到端:分块 prefill(边界切在图像区
                                          #   内部)下与 HF 逐 token 一致
python tests/parallel_check.py            # tp/pp/dp 布局对照单卡
python tests/async_check.py               # 异步调度器输出等价性
```

`tests/smoke_check_qwen35.py`(随机权重、宽松阈值)亦可运行;其通过率未因本工作
改变。

## Benchmark:与 vLLM 正面对比

同一负载、同一 GPU、双方引擎各自最优调度配置(vLLM `async_scheduling`;
nano-vllm `continuous_batching + async_scheduling`),形状匹配的预热,3 次计时
取中位(各次差异 <2%)。用 `benchmarks/bench_vs_vllm.py` 复现。

**硬件/软件**:1× RTX 5090、torch 2.13 / cu130、vLLM 0.27.1、bf16、
temperature 0.6、`ignore_eos`、256 请求、`max_model_len` 4096。
每次 rep 的原始输出:`benchmarks/results/vs-vllm-rtx5090.log`。

**Qwen3.5-2B**(本 fork 移植的混合 GDN 模型 —— vLLM 跑的是手工调优的内核,
nano-vllm 是洁净室移植):

| 负载 | nano-vllm | vLLM 0.27.1 | nano / vLLM |
|---|---|---|---|
| prefill 密集 —— 2048-token prompt、输出 2 | 总吞吐 **62.5k** tok/s | 62.1k | **100.6%** |
| decode 密集 —— 64-token prompt、输出 256 | **21.9k** 总 / 17.5k 输出 tok/s | 20.1k / 16.0k | **108.9%** |
| 混合 serving —— 输入/输出 ~ U(100, 1024) | **24.1k** 总 / 11.7k 输出 tok/s | 23.1k / 11.2k | **104.3%** |

**Qwen3-0.6B**(稠密模型,上游代码路径 + 本 fork 的异步调度):

| 负载 | nano-vllm | vLLM 0.27.1 | nano / vLLM |
|---|---|---|---|
| 混合 serving —— 输入/输出 ~ U(100, 1024) | **23.6k** 总 / 11.4k 输出 tok/s | 23.5k / 11.4k | **100.4%** |
| decode 密集 —— 64-token prompt、输出 256 | **45.2k** 总 / 36.2k 输出 tok/s | 43.5k / 34.8k | **103.9%** |

解读:nano-vllm 现在在两个模型的所有负载区间上都胜过 vLLM。但并非一直如此:
混合吞吐在 varlen-GDN 与统一异步重构前只有 **9.5k tok/s(约 vLLM 的 41%)**;
之后 prefill 落后 2%,靠一次 profile 找到 fla 输入守卫的拷贝链;最后的混合
劣势(98.7%)追溯到混合批在 CUDA graph 之外跑的 eager decode 卷积链 ——
三个故事都在上面的优化小节里。

```bash
OMP_NUM_THREADS=8 python benchmarks/bench_vs_vllm.py            # 双引擎,mixed
OMP_NUM_THREADS=8 python benchmarks/bench_vs_vllm.py --mode decode
OMP_NUM_THREADS=8 python benchmarks/bench_vs_vllm.py --mode prefill
# 各模式的负载定义见脚本头部注释
```

更多基准:`benchmarks/bench1.py`(nano 单独)、`benchmarks/bench.py`(vLLM
基线)、`benchmarks/bench_parallel.py`(tp/pp/dp 扩展性)。

## 许可

MIT —— 上游 © 2025 Xingkai Yu([GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm));
本 fork 的修改部分 © 2026 sileaver。
