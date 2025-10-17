# Lecture 2: Pytorch, Resource Accounting

两种资源：

* Memory(GB)
* Compute(FLOPs)

**Question**: How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?

​    total_flops = 6 * 70e9 * 15e12  # @inspect total_flops

​    assert h100_flop_per_sec == 1979e12 / 2

​    mfu = 0.5

​    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  # @inspect flops_per_day

​    days = total_flops / flops_per_day  # @inspect days

这里的70B指的是参数个数，如果算参数内存的话，还要乘以4 + 4 + (4 + 4)。

**Question**: What's the largest model that can you can train on 8 H100s using AdamW (naively)?

​    h100_bytes = 80e9  # @inspect h100_bytes

​    bytes_per_parameter = 4 + 4 + (4 + 4)  # parameters, gradients, optimizer state  @inspect bytes_per_parameter

​    num_parameters = (h100_bytes * 8) / bytes_per_parameter  # @inspect num_parameters

关于每个参数需要多少字节的“常驻状态”：权重本身+反向传播的梯度+Adam/AdamW 的两套一阶与二阶动量（m、v）

## Memory accounting

Almost everything (parameters, gradients, activations, optimizer states) are stored as floating point numbers.

* parameters：参数张量。刚开始模型加载阶段，存的只有这部分内存。
* activations：激活值张量。存的是中间结果，用于反向传播时计算梯度，大概是参数显存的$1 \sim 3$倍。并且随着 batch size / sequence length / layer 数 增长呈线性或平方级增长。
* gradients：梯度张量，只存在于训练阶段。约等于参数显存。
* optimizer states：更新张量，只存在于训练阶段。不同优化器，占用内存不同（SGD一倍，Adam大约2倍）。

### float32

最常见的/默认就是**float32**，四个字节。

![img](https://stanford-cs336.github.io/spring2025-lectures/images/fp32.png)

```python
x = torch.zeros(4, 8)  # @inspect x
assert x.dtype == torch.float32  # Default type
assert x.numel() == 4 * 8
assert x.element_size() == 4  # Float is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
```

## Compute accounting

### tensors on gpus

通常，tensor存储在CPU内存中，需要把它们移动到GPU内存。

```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")
```

![img](https://stanford-cs336.github.io/spring2025-lectures/images/cpu-gpu.png)

```python
if not torch.cuda.is_available():
    return
# GPU数量
num_gpus = torch.cuda.device_count()  # @inspect num_gpus
# 每个GPU信息
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)  # @inspect properties
# 已分配的内存
memory_allocated = torch.cuda.memory_allocated()  # @inspect memory_allocated
# 移动至GPU
text("Move the tensor to GPU memory (device 0).")
y = x.to("cuda:0")
assert y.device == torch.device("cuda", 0)
# 创建变量时直接指定GPU
text("Or create a tensor directly on the GPU:")
z = torch.zeros(32, 32, device="cuda:0")
new_memory_allocated = torch.cuda.memory_allocated()  # @inspect new_memory_allocated
memory_used = new_memory_allocated - memory_allocated  # @inspect memory_used
assert memory_used == 2 * (32 * 32 * 4)  # 2 32x32 matrices of 4-byte floats
```

### tensor operations

tensor storage

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-97aa05a6701b46521cb8a7c1e096c7e7-https_martinlwx_github_io_img_2D_tensor_strides_png)

```python
x = torch.tensor([
    [0., 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
])
assert x.stride(0) == 4
assert x.stride(1) == 1
r, c = 1, 2
index = r * x.stride(0) + c * x.stride(1)  # @inspect index
assert index == 6
```

tensor slicing: 切片、索引

tensor elementwise: 元素级操作、返回新tensor

tensor_matmul: y = x @ w

### tensor einops

简洁明了的维度操作。

### tensor operations flops

* 普通加法、乘法的浮点计算量等于矩阵大小。

* 矩阵乘法的浮点计算量是矩阵大小的两倍（元素相乘、相加）

* mfu = (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]

- FLOP/s depends on hardware (H100 >> A100) and data type (bfloat16 >> float32)

### gradients flops

Putting it togther:    

* Forward pass: 2 (# data points) (# parameters) FLOPs
* Backward pass: 4 (# data points) (# parameters) FLOPs
* Total: 6 (# data points) (# parameters) FLOPs

## Models    

可复现：设置随机数种子

