# Lecture 1: Overview and Tokenization

这门课是干什么的？

It's all about efficiency.

Resources: data + hardware (compute, memory, communication bandwidth)    

**How do you train the best model given a fixed set of resources?**

Example: given a Common Crawl dump and 32 H100s for 2 weeks, what should you do?

Design decisions:

![image-20251005193948096](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251005193948096.png)

## basics

### Tokenization

Tokenizers convert between strings and sequences of integers (tokens)

![image-20251005194130940](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251005194130940.png)

This course: Byte-Pair Encoding (BPE) tokenizer

### Architecture

Starting point: original Transformer.

![image-20251005194243552](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251005194243552.png)

Variants:

* Activation functions: ReLU, SwiGLU
* Positional encodings: sinusoidal, RoPE
* Normalization: LayerNorm, RMSNorm
* Placement of normalization: pre-norm versus post-norm
* MLP: dense, mixture of experts
* Attention: full, sliding window, linear
* Lower-dimensional attention: group-query attention (GQA), multi-head latent attention (MLA) 
* State-space models: Hyena

### Training

* Optimizer (e.g., AdamW, Muon, SOAP) 
* Learning rate schedule (e.g., cosine, WSD) 
* Batch size (e..g, critical batch size) 
* Regularization (e.g., dropout, weight decay)
* Hyperparameters (number of heads, hidden dimension): grid search

### Assignment 1

* Implement BPE tokenizer
* Implement Transformer, cross-entropy loss, AdamW optimizer, training loop
* Train on TinyStories and OpenWebText
* Leaderboard: minimize OpenWebText perplexity given 90 minutes on a H100 

## systems

Goal: squeeze the most out of the hardware

Components: kernels, parallelism, inference

### kernels

GPU = 流式多处理器（SM）+内存（Memory）

- 一个 GPU 由多个SM组成；
- 每个SM内包含多个CUDA 核心（CUDA cores）；
- 每个CUDA 核心执行线程的标量运算；
- SM 负责管理这些核心的线程调度、寄存器分配、共享内存访问等。

SM 的内部结构：

| 组件                              | 功能                                 |
| --------------------------------- | ------------------------------------ |
| CUDA cores（标量核心）            | 执行基础算术逻辑操作（如加法、乘法） |
| Tensor Cores（张量核心）          | 用于矩阵乘法和深度学习加速           |
| Warp Scheduler（线程束调度器）    | 管理线程束（warp）执行顺序           |
| Register File（寄存器文件）       | 存储每个线程的局部变量               |
| Shared Memory（共享内存）         | SM 内部线程共享的高速缓存            |
| Load/Store Units（加载/存储单元） | 负责与全局显存（global memory）交互  |
| SFU（特殊功能单元）               | 执行复杂函数，如 sin、cos、sqrt      |

GPU 执行一个 kernel 时：

1. kernel 被分成许多线程块（thread blocks）；
2. 每个线程块被分配到一个 SM；
3. SM 将线程分组成warp（线程束，通常 32 个线程）；
4. 调度器以 warp 为单位执行指令，隐藏内存访问延迟，实现高吞吐。

What a GPU (A100) looks like:

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-672bd77c57df485d07926615162a44d5-https_miro_medium_com_v2_resize_fit_2000_format_webp_1_6xoBKi5kL2dZpivFe1-zgw_jpeg)

Analogy: warehouse : DRAM :: factory : SRAM

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-27f5a4326a831bfe8b3af774827dd675-https_horace_io_img_perf_intro_factory_bandwidth_png)

Trick: organize computation to maximize utilization of GPUs by minimizing data movement

Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens

### parallelism

如果有多个GPU怎么办·？

GPU 之间的数据传输更慢，但同样遵循“最小化数据移动”的原则。

使用集合操作（例如：gather、reduce、all-reduce）。

在多个 GPU 之间分割：参数、激活值、梯度和优化器状态。

计算的划分方式包括：数据并行（data parallelism）、张量并行（tensor parallelism）、流水线并行（pipeline parallelism）和序列并行（sequence parallelism）。

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-f92abaa9bd1dcd5fc90c7b74f3b29260-https_www_fibermall_com_blog_wp-content_uploads_2024_09_the-hardware-topology-of-a-typical-8xA100-GPU-host_png)

### inference

Goal: generate tokens given a prompt (needed to actually use models!)

Inference is also needed for reinforcement learning, test-time compute, evaluation

Globally, inference compute (every use) exceeds training compute (one-time cost)

Two phases: prefill and decode

#### Prefill 阶段

**作用：处理提示（prompt）输入。**

- 在这一阶段，模型接收整段输入文本（prompt）。
- 模型需要一次性计算所有输入 token 的表示（embeddings），并通过 Transformer 层建立上下文依赖。
- 输出结果是：
  - 最后一个 token 的隐藏状态embedding（用于预测下一个 token）
  - 所有中间层的Key/Value 缓存（KV cache）

这些缓存会在下一阶段（decode）中复用，避免重复计算。

**特点：**

- 计算量大，因为是并行处理整个输入序列。
- 延迟高（first-token latency），因为必须等所有输入都计算完才能输出第一个预测。

#### Decode 阶段

**作用：逐步生成新 token。**

- 每生成一个新的 token，模型会：
  1. 取出上一步的 KV cache；
  2. 只对新 token进行前向计算；
  3. 更新 KV cache；
  4. 输出下一个 token 的概率分布。
- **该过程会循环多次，直到生成结束。**

**特点：**

- 每次只处理一个（或少量）token，计算量相对较小。
- 能部分并行（如多样本 batch 或 speculative decoding）。
- 延迟主要来自每步循环（token-by-token generation）。

![img](https://stanford-cs336.github.io/spring2025-lectures/images/prefill-decode.png)

​    

Methods to speed up decoding:

* Use cheaper model (via model pruning, quantization, distillation)
* Speculative decoding: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)
* Systems optimizations: KV caching, batching

### Assignment 2

* Implementa fused RMSNorm kernel in Triton
* Implement distributed data parallel training
* Implement optimizer state sharding
* Benchmark and profile the implementations

## scaling laws

Goal: do experiments at small scale, predict hyperparameters/loss at large scale

Question: given a FLOPs budget ($C$), use a bigger model ($N$) or train on more tokens ($D$)?

![img](https://stanford-cs336.github.io/spring2025-lectures/images/chinchilla-isoflop.png)

得出的经验： $D^* = 20 N^*$ (e.g., 1.4B parameter model should be trained on 28B tokens)

But this doesn't take into account inference costs!

### Assignment 3

We definea training API (hyperparameters -> loss) based on previous runs

Submit "training jobs" (undera FLOPs budget) and gather data points

Fita scaling law to the data points

Submit predictions for scaled up hyperparameters

Leaderboard: minimize loss given FLOPs budget

## data

Question: What capabilities do we want the model to have?

Multilingual? Code? Math?

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-b3aebfa83a900cd491e70acf27806db3-https_ar5iv_labs_arxiv_org_html_2101_00027_assets_pile_chart2_png)

### Data curation

Data does not just fall from the sky.

* Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc.
* Appeal to fair use to train on copyright data? 
* Might have to license data (e.g., Google with Reddit data) 
* Formats: HTML, PDF, directories (not text!)

### Data processing

* Transformation: convert HTML/PDF to text (preserve content, some structure, rewriting)
* Filtering: keep high quality data, remove harmful content (via classifiers)
* Deduplication: save compute, avoid memorization; use Bloom filters or MinHash

### Assignment 4

* Convert Common Crawl HTML to text
* Train classifiers to filter for quality and harmful content
* Deduplication using MinHash
* Leaderboard: minimize perplexity given token budget

## alignment

So far, a **base model** is raw potential, very good at completing the next token.

Alignment makes the model actually useful.

Goals of alignment:

* Get the language model to follow instructions
* Tune the style (format, length, tone, etc.)
* Incorporate safety (e.g., refusals to answer harmful questions)

**Two phases:**

* supervised finetuning
* learning from feedback

### SFT

Instruction data:  (prompt, response) pairs

Supervised learning: fine-tune model to maximize p(response | prompt).

### Preference data

Now we have a preliminary instruction following model.

Let's make it better without expensive annotation.

Data: generate multiple responses using model (e.g., [A, B]) to a given prompt.

User provides preferences (e.g., A < B or A > B).

```
preference_data: list[PreferenceExample] = [
 PreferenceExample(
     history=[
         Turn(role="system", content="You are a helpful assistant."),
         Turn(role="user", content="What is the best way to train a language model?"),
     ],
     response_a="You should use a large dataset and train for a long time.",
     response_b="You should use a small dataset and train for a short time.",
     chosen="a",
 )
]
```

### Verifiers

* Formal verifiers (e.g., for code, math)

* Learned verifiers: train against an LM-as-a-judge

### Algorithms

* Proximal Policy Optimization (PPO) from reinforcement learning
* Direct Policy Optimization (DPO): for preference data, simpler
* Group Relative Preference Optimization (GRPO): remove value function

### Assignment 5

* Implement supervised fine-tuning
* Implement Direct Preference Optimization (DPO)
* Implement Group Relative Preference Optimization (GRPO)

## Byte Pair Encoding (BPE)

Basic idea: *train* the tokenizer on raw text to automatically determine the vocabulary.

Intuition: common sequences of characters are represented by a single token, rare sequences are represented by many tokens.

```
def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
 indices = list(map(int, string.encode("utf-8")))  # @inspect indices
 merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
 vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
 for i in range(num_merges):
     counts = defaultdict(int)
     for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
         counts[(index1, index2)] += 1  # @inspect counts
     pair = max(counts, key=counts.get)  # @inspect pair
     index1, index2 = pair
     new_index = 256 + i  # @inspect new_index
     merges[pair] = new_index  # @inspect merges
     vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
     indices = merge(indices, pair, new_index)  # @inspect indices
 return BPETokenizerParams(vocab=vocab, merges=merges)
```





