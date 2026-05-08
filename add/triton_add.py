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