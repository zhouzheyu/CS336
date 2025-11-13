# Lecture 8:  Parallelism2

跨GPU通信：可能当前GPU需要的数据存在其他GPU中。

fusion tiling：与其在HBM读写数据，不如直接加载到L1 cache/shared memory，在本地暂存区处理，然后再写会HBM

Generalized hierarchy (from small/fast to big/slow):

* Single node, single GPU: L1 cache / shared memory
* Single node, single GPU: HBM
* Single node, multi-GPU: NVLink
* Multi-node, multi-GPU: NVSwitch

## Part1: building blocks of distributed communication/computation

### collective operations

#### Broadcast

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-525847c9d4b48933cb231204a2d13e0e-https_pytorch_org_tutorials__images_broadcast_png)

#### Scatter

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-3aa3584628cb0526c8b0e9d02b15d876-https_pytorch_org_tutorials__images_scatter_png)

#### Gather
![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-7e8670a3b7cdc7848394514ef1da090a-https_pytorch_org_tutorials__images_gather_png)

#### Reduce

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-1c451df4406aea85e640d1ae7df6df31-https_pytorch_org_tutorials__images_reduce_png)

#### All-Gather

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-4a48977cd9545f897942a4a4ef1175ac-https_pytorch_org_tutorials__images_all_gather_png)

#### Reduce-Scatter

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-66ea136cfe7f3e7394fd0b056fd9d949-https_docs_nvidia_com_deeplearning_nccl_user-guide_docs__images_reducescatter_png)

#### All-reduce = reduce-scatter + all-gather

![img](https://stanford-cs336.github.io/spring2025-lectures/var/files/image-0ef9693f0008d5a75aa5ac2b542b83ac-https_pytorch_org_tutorials__images_all_reduce_png)

### torch distributed

use PyTorch distributed library (**torch.distributed**)

* Provides clean interface for collective operations (e.g., all_gather_into_tensor)
* Supports multiple backends for different hardware: gloo (CPU), nccl (GPU)
* Also supports higher-level algorithms (e.g., FullyShardedDataParallel) [not used in this course]

## Part 2: distributed training

* data_parallelism()         # Cut up along the batch dimension
* tensor_parallelism()       # Cut up along the width dimension
* pipeline_parallelism()     # Cut up along the depth dimension

以拼接若干个MLP为例。

**MLP是transformer的性能瓶颈，而不是attention**

### 前置工作

```python
import torch.distributed as dist

def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
def cleanup():
    torch.distributed.destroy_process_group()
```

### data parallelism

重点在于对grad的all reduce同步操作。

```python
for param in params:
    dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
```

完整参考代码：

总共有world size个进程运行这这个相同的函数（下同）

```python
import torch.distributed as dist

def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)
    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(get_device(rank))
    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state
    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude
        # Backward pass
        loss.backward()
        # Sync gradients across workers (only difference between standard training and DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
        # Update parameters
        optimizer.step()
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)
    cleanup()
```

### tensor parallelism

重点在于activation后的all reduce / all gather同步操作。

```python
# Allocate memory for activations (world_size x batch_size x local_num_dim)
activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]
# Send activations via all gather
dist.all_gather(tensor_list=activations, tensor=x, async_op=False)
```

完整参考代码：

从代码中也可以看出，TP同步通信的频率很高。

```python
def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  @inspect local_num_dim
    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]
    # Forward pass
    x = data
    for i in range(num_layers):
        # Compute activations (batch_size x local_num_dim)
        x = x @ params[i]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)
        # Allocate memory for activations (world_size x batch_size x local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]
        # Send activations via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)
        # Concatenate them to get batch_size x num_dim
        x = torch.cat(activations, dim=1)
    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)
    # Backward pass: homework exercise
    cleanup()
```

### pipeline parallelism

```python
def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)
    # Use all the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    # Split up layers
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers
    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]
    # Forward pass
    # Break up into micro batches to minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]
    for x in micro_batches:
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)
        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)
        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)
            
		# Backward pass: homework exercise
		
		cleanup()
```

Not handled: overlapping communication/computation to eliminate pipeline bubbles

##     Summary

* Many ways to parallelize: data (batch), tensor/expert (width), pipeline (depth), sequence (length)
* Can **re-compute** or store in **memory** or store in another GPUs memory and **communicate**

* Hardware is getting faster, but will always want bigger models, so will have this hierarchical structure







