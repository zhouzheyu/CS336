# Lec4: Mixture of Experts

## 基本介绍

![image-20251013171058814](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013171058814.png)

优点：

* 相比dense layer，MoE稀疏激活experts，让模型参数更多但训练推理的FLOPs开销基本不变，训练更快，效果更好。
* MoE很方便实现设备并行：每个expert被部署在一个设备上。同时由于稀疏激活的特征，所以你只需要获取您的token，并将其路由到适当的设备上，然后计算就会在该设备上执行。在各个expert输出结果，再带回之前的地方。

缺点：

* 通信和调度非常复杂混乱。只有当进行多结点训练的时候，MoE的最大优势才真正显现出来。当你必须拆分你的模型，将你的experts放到不同节点的时候，才是有意义的。
* 路由选择哪个expert，是很难决定的。
* 路由选择的训练并不是可微分的，这一目标的训练，要么是启发式的，要么是不稳定的。
  * 所谓“启发式”就是指这些设计更多是试探性和经验性的，没有严格的理论最优性证明。比如负载均衡损失（load balancing loss）或路由正则化项，用来鼓励路由器均匀使用专家、避免塌缩。

![image-20251013172954376](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013172954376.png)

## Routing function

最常用的是Token chooses expert

![image-20251013173358935](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013173358935.png)

![image-20251013173539457](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013173539457.png)

![image-20251013173603043](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013173603043.png)

### Top-K routing

K > 1，对输出结果加权求和。

softmax实际上的作用是归一化。

要topK而不是N，是为了训练和推理的效率，保证experts被稀疏激活。

有的是在topK之后softmax。

一般都会用残差结构，保证梯度流恒定。

![image-20251013173714640](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013173714640.png)

### variations

* Conventional Top-2 Routing
  * K = 2，不仅会关注性能最好的，同时也会探索第二支arm的潜在可能。此时FLOPs就会翻一倍。
* Fine-grained Expert Segmentation
  * Fine-grained ratio：每个expert会被切分成多少个小expert，维度为原来的$1 / m$。
* Shared Expert Isolation (DeepSeekMoE)
  * 需要shared FFN来处理所有的token的公共知识。
* 根据activated FFN的数量，和每个FFN size大小，换算成对应dense layer的FLOPs的多少倍。

![image-20251013174056419](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013174056419.png)

![image-20251013175624129](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013175624129.png)

## Training objectives

稀疏激活experts的决策并不可微分。

### RL to optimize gating policies

RL is the ‘right solution’ but gradient variances and complexity means it’s not widely used.

![image-20251013175826727](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013175826727.png)

### Stochastic perturbations

解决 MoE 在路由器训练不稳定、梯度估计困难、负载不平衡等问题。

加入噪声依旧不可微分，但在梯度下降的时候会引入一些信号。

![image-20251013180623775](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013180623775.png)

### Heuristic/loading balancing losses

我们希望系统能够均匀地使用experts，保证训练效率。

* $f_i$：expert $i$ 分配得到的token的百分比。
* $P_i$：router 分配给 expert $i$ 的概率期望。

![image-20251013181416252](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013181416252.png)

![image-20251013181518989](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013181518989.png)

#### per-expert biases

DeepSeek v3 variation采用的方法。

如果我们放弃其他限制，那么出现的大问题就是，最终只会选择一位expert，什么都很擅长，最后到达局部最小值，所有的token都会选择这个expert。浪费很多experts和内存，同时loss也会更大，效率更低。

没得到足够token，$b_i$更高，更有希望选择这个。

![image-20251013181532544](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013181532544.png)

#### 实验效果

路由experts更均匀，每个token都能找到更合适自己的expert，loss更小，训练效果更好。

![image-20251013182011359](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013182011359.png)

MoE can parallellize device nicely.

![image-20251013220945619](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013220945619.png)

![image-20251013221036657](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013221036657.png)

## issues

### 结果随机

MoE models具有随机性，路由（routing）和容量计算都是在整个 batch 维度上进行的，如果很多其他人的token和我的都路由到同一个FFN，而当前FFN所在device的内存已满，意味着就将从这个batch中丢弃已选中的token。

![image-20251013224203875](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013224203875.png)

### softmax不稳定

训练loss下降不稳定，最好别用softmax，改用z-loss。

![image-20251013224632154](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013224632154.png)

![image-20251013224644467](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013224644467.png)

### fine tune过拟合

由于MoE的稀疏激活特点，很容易在小数据集的微调上过拟合。

解决方法：

* MoE和dense交叉使用
* 使用更大的数据集

![image-20251013224855530](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013224855530.png)

## upcycling

不用从头训练 MoE，而是直接用一个已训练好的 dense LM语言模型来初始化它。

* 复制 Dense FFN 参数到所有 experts
* 随机初始化 Router（gating network）
* 继续训练

![image-20251013225434312](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251013225434312.png)

## DeepSeek MoE

### DeepSeek V1

![image-20251014115816190](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251014115816190.png)

### DeepSeek V2

Top-M device routing: 

当experts太多，路由到这些experts的通信成本更高，如果太过分散，必须向很多设备发送很多token。

先从设备粒度做一个限制，再在那些设备内选专家。

每个 token 的 hidden state $h_t$ 输入一个线性路由器得到 logits，形状为 `[num_devices]`，表示 token 对每个设备的打分。形式上：$s_t = W_d \cdot h_t$。对这些 logits 做 softmax 或加噪声后取 Top-M 最大的设备。

Communication balancing loss – balancing both communication in and out: 

即便采用 Top-M Device Routing 限制 token 只路由到部分设备，也还可能出现这样的不平衡情况：

- 某些设备收到 token 较多（incoming traffic 较重），成为通信瓶颈；
- 虽然设备被选中次数可能均匀，但传入 / 传出 token 的数量可能极不均匀；
- 发生这种通信不平衡，就会产生延迟、带宽拥堵或通信热点，影响整体吞吐率。

![image-20251014120016097](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251014120016097.png)

### DeepSeek V3

* 用Sigmoid对路由选择做归一化，然后用Softmax计算experts权重
* per-expert bias

![image-20251014121311523](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251014121311523.png)

## MLA : Multihead Latent Attention

express the Q, K, V as functions of a lower-dim, ‘latent’ activation

但是不适用于RoPE，因为RoPE是在 $Q$ 和 $K$ 加入位置信息。

![image-20251014121406053](/Users/zhouzheyu/Library/Application Support/typora-user-images/image-20251014121406053.png)

