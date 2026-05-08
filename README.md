# BasicKernels：常见的算子实现【cuda 和 cann】

下面是对于常见的算子与其代码实现，包括朴素版本和优化实现

| 算子            | 描述         |
| --------------- | :----------- |
| add             | 张量加       |
| concat          | 拼接         |
| relu            | 激活函数     |
| softmax         | 激活函数     |
| gemv            | 矩阵向量乘法 |
| gemm            | 矩阵乘法     |
| conv2D          | 卷积乘       |
| RSMnorm         | 正则化       |
| Reduce          | 归约         |
| Transpose       | 转置         |
| attention       | 注意力       |
| flash attention | 注意力优化   |
| mxfp8           | 量化         |
| mxfp4           | 量化         |


...

## 1. elementwise

elementwise指的是逐元素进行操作，最常见包括add，concat，relu等。这些都是是访存密集型算子，计算强度低，性能受限于GPU显存带宽。

1.1 cuda实现

```
__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    // 计算当前线程处理的元素索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;
  
    // 确保线程不会越界访问数组
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

// 启动内核
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
```

对于cuda的线程模型

```
Grid
├── Block 0 (256 threads)
│   ├── Thread 0 → 处理元素 0
│   ├── Thread 1 → 处理元素 1
│   └── ...
├── Block 1 (256 threads)
└── ...
```

1.2 triton实现

```
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    # 获取当前程序的ID（类似 CUDA 中的 blockIdx）
    pid = tl.program_id(axis=0)
  
    # 计算当前 block 处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
  
    # 创建掩码，防止越界
    mask = offsets < n
  
    # 加载数据（自动处理边界）
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
  
    # 计算
    c = a + b
  
    # 存储结果
    tl.store(c_ptr + offsets, c, mask=mask)


def vector_add(a, b):
    n = a.shape[0]
    c = torch.empty_like(a)
  
    # 自动计算需要的 block 数量
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # 等价于 (n + BLOCK_SIZE - 1) // BLOCK_SIZE
  
    vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)
    return c


def main():
    # 测试参数
    N = 1024
  
    # 在 CPU 上创建随机数据
    a_cpu = torch.randn(N, dtype=torch.float32)
    b_cpu = torch.randn(N, dtype=torch.float32)
  
    # 将数据移到 GPU
    a = a_cpu.cuda()
    b = b_cpu.cuda()
  
    # 调用 Triton kernel
    c = vector_add(a, b)
  
    # CPU 计算结果作为对照
    c_ref = a_cpu + b_cpu
  
    # 验证结果
    if torch.allclose(c.cpu(), c_ref, atol=1e-5):
        print(f"✅ 测试通过! N={N}")
        print(f"   前5个结果: {c[:5].cpu().tolist()}")
    else:
        print(f"❌ 测试失败!")
        diff = (c.cpu() - c_ref).abs().max()
        print(f"   最大误差: {diff}")


if __name__ == "__main__":
    main()

```

对于triton来说

```
Grid
├── Program 0 (处理 256 个元素，一次性向量化)
├── Program 1 (处理 256 个元素，一次性向量化)
└── ...
```



1.3 cutluss 实现（不推荐）
