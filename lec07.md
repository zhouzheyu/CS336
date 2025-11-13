# Lecture 7:  Parallelism1

What we want from multi-machine scaling:

* Linear memory scaling (max model params scales with num gpus)
* Linear compute scaling (model flops scale linearly with num gpus)

batch size是做 parallelism 很重要的一个因素。

## Basics

一些分布式通信原语。

* All-Reduce：把所有 rank 的输入数据做**归约**（比如求和、求平均等），然后把结果**广播**给所有 rank。

* All-Gather：每个 rank **收集**其他rank的输入数据，让每个 rank 都获得所有这些数据的完整集合。
* Reduce-Scatter：对所有 rank 的输入数据执行**归约**操作（例如求和、平均等），然后把结果**分片**给不同的 rank。

![image-20251111182133057](media/7-1.png)

All-Reduce = Reduce-Scatter + All-Gather。

![image-20251111183145548](media/7-2.png)

## Data parallelism

* split the elements of B sized batch across M machines. 
* **Exchange gradients to synchronize.**

### Naïve data parallel

How does this do?

* Compute scaling – each GPU gets B/M examples.
* Communication overhead – transmits 2x # params every batch（All-Reduce操作，每个GPU先发自己计算的梯度进行reduce，然后broadcast到各个GPU）. OK if batches are big
* Memory scaling – none. Every GPU needs # params at least

Problem：

*  we copy the model parameters to each GPU.	

![image-20251111212319444](media/7-6.png)

### ZeRO levels 1-3

纵轴表示的是model parameters

![image-20251111211316351](media/7-3.png)

#### ZeRO stage 1

在各个GPU分割optimizer state。

* 因为每个GPU只会存部分optimizer state，因此把gradients ReduceScatter到各个GPU。
* 每个GPU只根据它存的那部分optimizer state上的gradients。
* 最后all gather各自更新的gradients（带来额外的通信开销）

![image-20251111211535726](media/7-4.png)

![image-20251111212126124](media/7-5.png)

![image-20251111212415771](media/7-7.png)

#### ZeRO stage 2

分割gradients到各个GPU。

- 每算完一层，就马上把这层梯度通过 **Reduce-Scatter** 传给负责这一分片的 GPU；
- 然后立即释放本地梯度显存。
- 算完所有层，就知道了所有参数的梯度，并且存在各个GPU上。

![image-20251111212556138](media/7-8.png)

![image-20251111213251331](media/7-9.png)

#### ZeRO stage 3 (FDSP)

在各个GPU分割model parameter。

* Forward pass的时候All-Gather model parameter。
* 两次All-Gather：Forward pass + update paramter
* 一次ReduceScatter：算gradients

![image-20251111213411920](media/7-10.png)

![image-20251111214757428](media/7-11.png)

![image-20251111214604690](media/7-14.png)

### Issues

缺点：

* 最多分割到Batch size个GPU。
* 引入了显著的通信和性能开销，速度更慢。
* ZeRO并没有处理 activation ，激活值内存过大时出现问题。

优点：

* 数据并行并不关注架构。

![image-20251111214348159](media/7-12.png)

![image-20251111214505316](media/7-13.png)

## Model parallelism

| 对比项              | 模型并行 (Model Parallelism)                        | ZeRO Stage 3           |
| ------------------- | --------------------------------------------------- | ---------------------- |
| 切分的内容          | 模型参数                                            | 参数 + 优化器状态      |
| 每个 GPU 计算的范围 | 只计算模型的一部分                                  | 计算整个模型           |
| GPU通信的内容       | 激活值 (activations)                                | 参数和梯度             |
| 代表性例子          | Tensor / Pipeline Parallelism（Megatron-LM、GPipe） | DeepSpeed ZeRO Stage 3 |

### Layer-wise parallel

![image-20251112161259469](media/7-15.png)

很多GPU处于空闲，吞吐量只相当于单个GPU。 

![image-20251112161401965](media/7-16.png)

### Pipeline parallel

解决Layer-wise parallel的低吞吐问题。

* Pipelines 相对 Data Parallel 更省显存
  * Data Parallel每个 GPU 都保存整份模型参数、优化器状态和激活值；虽然通过梯度 all-reduce 实现了并行训练，但显存开销是重复的。
  * Pipeline Parallel模型被切分到不同 GPU，每个 GPU 的参数内存显著减少；激活值只需保留当前阶段需要的部分；优化器状态也只属于自己那一部分参数。
* Pipelines can have good communication properties (compared to FDSP) – it depends only on activations ($b \times s \times h$) and is **point to point**.
  * 每个阶段（GPU）只与**前一个和后一个阶段**通信（point-to-point）；
  * 通信内容是**激活值 (activations)**，大小大约为：$b \times s \times h$
  * 这比起 DDP 或 tensor parallel 中传递**整层参数或梯度**（大小为参数量）要**小得多**，而且通信是点对点（P2P），不需要全局同步（如 All-Reduce）。
* Batch size很重要！
* Generally, we will use pipelines on **slower network links** (i.e. inter-node) as a way to get better memory-wise scaling.

![image-20251112161944780](media/7-17.png)

![image-20251112163359216](media/7-18.png)

### Tensor parallel

identity：无需通信的。

由于TP的高频通信要求，常在一台服务器内的GPU跑。

**$f$ 和 $g$ 都是 synchronize barria，最后要做一次All-Reduce。**

![image-20251112164318869](media/7-19.png)

![image-20251112164559061](media/7-20.png)

![image-20251112171913898](media/7-30.png)

![image-20251112164800599](media/7-21.png)

## Activation parallelism

### A final complexity - activation memory

![image-20251112165017217](media/7-22.png)

![image-20251112165824877](media/7-23.png)

![image-20251112165925865](media/7-24.png)

![image-20251112165956748](media/7-25.png)

### Sequence parallel

LayerNorm 和 Dropout 层不依赖序列之间的关联，适合做SP。

**reduce-scatter -- SP -- All-Gather。**

![image-20251112171150282](media/7-26.png)

![image-20251112171451372](media/7-27.png)

## Recap

![image-20251112171617466](media/7-28.png)

![image-20251112171719629](media/7-29.png)

![image-20251112171951196](media/7-31.png)

还有当下一些主流LLM的并行策略...看讲义。













