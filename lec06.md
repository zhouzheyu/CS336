# Lecture6: Kernels, Triton

学习为GPU编写高性能代码。

warp的作用：减少了控制机制的数量，因为这些线程都是同时执行的，不需要为每个线程都配备一个控制机制。

## benchmarking and profiling

### benchmarking

基准测试和性能分析，总共花费多少时间/时间被用在哪些地方。找到瓶颈模块

warmup的作用：确保没有测量启动时间（编译时间）。

torch.cuda.synchronize：python code存在CPU中，当运行的时候，会将一堆cuda kernel分配给GPU，GPU会执行这些操作，而CPU实际上会继续执行后续内容。确保CPU和GPU处于相同状态，并且没有排队运行的东西。对正在执行的代码而言，处于同一点。

这对写高性能代码很好，但是当做benckmark和profile的时候，因为此时GPU在跑模型，但CPU在做其他的事，实际上并没有测量GPU的执行时间。

运行前后都要调用torch.cuda.synchronize。

因为GPU发热，可能会有误差，测量多次，然后取平均值。

### benchmarking 矩阵乘法

1024-2048:执行这些矩阵乘法的时候存在恒定因子开销。这些数字必须从CPU传到GPU，启动内核将产生开销。

### profiling

更好地了解程序在硬件上的实际执行情况。查看哪个cuda kernel被调用，不同的matrix size，调用的cuda kernel不同。

接口占用了大量的CPU时间，以及它们在向GPU发送数据操作时产生的开销。

PyTorch有很多内置的profiler。

100%：因为没有GPU

有的时候硬件不同、矩阵大小不同，...也不同。

torch compile：有一个选项对硬件上的矩阵乘法进行micro benchmark，然后为模型选择性能最高的matrix multiply subroutines。可以为模型提速10%。

最多显示10个。

CPU运行的更快，如果打印的话，说明CPU必须要等GPU计算出当前层的损失。

瓶颈一直都是GPU：CPU可以提前运行并将命令排队到CPU中。CPU和GPU是分开的。

## kernel fusion motivation

Gelu kernel, 发给gpu，进行计算，然后返回结果

Wrapper,驻足在cpu上，协调kernel的启动。

__global

![image-20251107183927686](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251107183927686.png)

Block size大小的选择：内存块够多吗，够不够让SM满负荷运转。每个block的工作量是否足够。

数据不会一直存放在sm中，在global memory和sm之间传递。



## cuda kernels

write kernels in cuda/C++



## triton kernels

write kernels in python

更高级的抽象，不用管理gpu的方方面面

不用再考虑线程的问题

可以管理内存的合并

每个SM内需要停止或启动线程，这些都由triton自动管理

跨SM的调度需要手动操作

编程时站在SM的角度思考，编译器会处理更多底层细节。

性能上可以比很多PyTorch实现的高出不少

没有使用偏移量，因为我们没有编写线程程序，而是编写的模块化程序

offset是一个vector，not a single value.

这种向量化处理，将会被不同的线程处理

## python compilation

使用PyTorch现有的JIT编译器进行优化。

Don't write kernels at all?

PTX：非常接近机器代码，以了解当编写代码的时候，GPU在底层实际执行的操作。

 Torch.compile

将未经优化的代码编译成更优化的代码，自动优化，例如kernel fusion。

在底层生成triton。

有时候面对复杂任务可能torch.compile做的没那么好，当发现达不到预期的时候，还是要自己写triton

![image-20251107191251984](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251107191251984.png)

## triton softmax main

使用Trition实现softmax：more advanced computations

之前的都是元素级操作，但softmax有一个reduction操作

