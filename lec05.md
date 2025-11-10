# Lecture 5: GPUs

![image-20251028105420994](media/5-1.png)

❖ Part 1: GPUs in depth – how they work and important parts

❖ Part 2: Understanding GPU performance

❖ Part 3: Putting it together – unpacking FlashAttention

很多时候，算力的提升能带来语言模型性能的可预测性增强（Faster hardware, better utilization, improved parallelization）。

![image-20251028105930580](media/5-2.png)

## GPUs in depth

CPU保证的是低延迟，GPU保证的是高吞吐。

GPU有更多的计算单元（ALU），同时有更少的控制和缓存。

![image-20251028110342840](media/5-3.png)

### GPU结构

* 计算单元：若干个SM。一个SM又包含多个SP，分别执行一个线程。
* 存储器：距离SM越近，访问速度越快。
  * shared memory和L1 cache：SM内部
  * L2 cache：GPU芯片上
  * global memory：next to GPU

![image-20251028111246691](media/5-4.png)

![image-20251028111257265](media/5-5.png)

### Execution model

GPU使用的是SIMT（单指令多线程）架构。

* 每个 Block 会被分配在一个 SM 的 shared memory 执行，SM 内部再以 Warp 为单位调度线程。
* warp由若干个线程组成（通常32个），来自同一个 warp 的内存访问同时发生。
* 相同指令不同数据：一个 warp 会执行同一条指令，但每个线程处理自己的数据。
* warp的作用：减少了控制机制的数量，因为这些线程都是同时执行的，不需要为每个线程都配备一个控制机制。

![image-20251028112541848](media/5-6.png)

### Memory model

* Host：指的是CPU端的内存。
  * GPU 无法直接访问 CPU 内存，需要通过 PCIe 或 NVLink 通信。
  * Host 负责把数据传入 GPU 的 global memory，或者从中取出结果。
* Global Memory：不同Block之间共享数据必须经过global memory。
* Block：每个 Block通常由一个SM执行。内部包含：
  * Shared Memory：仅 block 内线程共享，block 外线程看不到。
  * Registers：每个线程**私有**的最快速存储，存放局部变量、临时计算值。
  * Threads：
    - 每个线程运行相同的 kernel 代码。
    - 可以访问自己的寄存器、当前 block 的 shared memory，以及整个 GPU 的 global memory。

![image-20251028113009720](media/5-7.png)

### Strengths of the GPU model

![image-20251028113125189](media/5-8.png)

![image-20251028113151290](media/5-9.png)

![image-20251028113750595](media/5-10.png)

### GPU的性能瓶颈：memory bandwidth

计算性能（FLOPs）增长得比内存带宽快，计算单元空闲时间更多。

![image-20251028113803484](media/5-11.png)

### Recap

![image-20251028115033061](media/5-12.png)

## make GPUs go fast

横轴：矩阵宽度

纵轴：FLOPs速度

![image-20251028115220715](media/5-13.png)

![image-20251028115319831](media/5-14.png)

### Control divergence

![image-20251028115557990](media/5-15.png)

### Low precision computation

存的字节数更少，需要移动的内存也就更少。

![image-20251028115719505](media/5-16.png)

### Operator fusion

通过Operator fusion减少内存的访问次数。

比如，自己写一个cuda kernal，中间结果没有太多依赖，融合多种单一操作，无需把所有内容发送回全局内存。这种非常简单的融合操作可以由编译器自动完成（torch compile）。

![image-20251028120824352](media/5-17.png)

![image-20251028120857286](media/5-18.png)

### Recomputation

使用更多的计算避免内存访问。在反向传播中动态地重新计算中间激活值。

原本的内存读写次数：

![image-20251028121057334](media/5-19.png)

不存中间激活值，重新计算：

![image-20251028121129330](media/5-20.png)

### Coalescing memory (burst model)

GPU的全局存储（DRAM）实际上非常非常慢。对DRAM做出的优化之一就是，当读取一段内存的时候，实际上不会只得到该值，得到的是a whole chunk of memory。这称之为burst mode。

当你寻址内存的时候，为了从内存中发信号，那些字节必须移动到放大器上，这是一个缓慢的步骤。burst mode实际上掩盖了存储位置移动到放大器上这一昂贵步骤。

Memory coalescing：同时可以获得burst section倍的内存访问吞吐量（只需访问首字节）。

![image-20251028122118435](media/5-21.png)

![image-20251028122154128](media/5-22.png)

![image-20251028122234111](media/5-23.png)

### Tiling

**将片段从全局内存加载到共享内存，合并内存访问，以尽量减少我们必须进行的全局内存访问量。**

这存在共享内存的tile可以重复读。

![image-20251028122328293](media/5-24.png)

![image-20251028122809829](media/5-25.png)

#### 学会优化tile size

![image-20251028122924148](media/5-26.png)

#### 对齐tile和burst section

建议tile或matrix size是burst size的整数倍。必须进行padding，获得良好的矩阵大小，实现对齐。

![image-20251028123051279](media/5-27.png)

### 小结

torch compile和cuda optimization，做的就是以上这些事，从而实现更好的表现。

![image-20251028123621215](media/5-28.png)

### Matrix mystery

![image-20251028123440421](media/5-29.png)

1536之前，没有足够多的矩阵乘法任务要做，因此加载矩阵和基本的IO就成了在此之前的性能瓶颈。

在此之后，吞吐量大幅下降，没有足够的内存带宽支持您的计算单元。

越是2的幂次方，越容易出现tile和burst section的对齐。

![image-20251028123505296](media/5-30.png)

1792：会有$7 \times 14$个tile。到1793之后，增加tile数量到$8 \times 15$，同时利用率降低，而且很难并行执行。

![image-20251028123526810](media/5-31.png)



希望在tile中做softmax，不要回到那个大矩阵

在反向传播的时候，同样不想存储n2级别的中间激活值，同样要一个tile一个tile地计算。



memory movement永远都是这一切的瓶颈，尽可能避免高带宽内存/全局内存的memory movement。

不要一直想着如何减少FLOPs的数量。
