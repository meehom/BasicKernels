import torch
import triton
import triton.language as tl


@triton.jit
def vector_concat_kernel(
    a_ptr, b_ptr, c_ptr,
    n1, n2,
    BLOCK_SIZE: tl.constexpr
):
    # 获取当前程序的ID（类似 CUDA 中的 blockIdx）
    pid = tl.program_id(axis=0)

    # 计算当前 block 处理的元素范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total = n1 + n2

    # 前半段拷贝 a，后半段拷贝 b
    mask_a = offsets < n1
    mask_b = (offsets >= n1) & (offsets < total)

    a_vals = tl.load(a_ptr + offsets, mask=mask_a)
    b_vals = tl.load(b_ptr + (offsets - n1), mask=mask_b)

    tl.store(c_ptr + offsets, a_vals, mask=mask_a)
    tl.store(c_ptr + offsets, b_vals, mask=mask_b)


def _check_inputs(a, b):
    assert a.is_cuda and b.is_cuda, "input tensors must be on CUDA"
    assert a.ndim == 1 and b.ndim == 1, "only 1D tensors are supported"
    assert a.dtype == b.dtype, "input dtypes must match"
    assert a.device == b.device, "input devices must match"

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    return a, b


def launch_vector_concat(a, b, c):
    n1 = a.shape[0]
    n2 = b.shape[0]
    block_size = 256
    grid = (triton.cdiv(n1 + n2, block_size),)
    vector_concat_kernel[grid](a, b, c, n1, n2, BLOCK_SIZE=block_size)


def vector_concat(a, b):
    a, b = _check_inputs(a, b)
    c = torch.empty((a.shape[0] + b.shape[0],), device=a.device, dtype=a.dtype)
    launch_vector_concat(a, b, c)
    return c


def benchmark(fn, warmup=10, runs=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        fn()
    stop.record()
    torch.cuda.synchronize()

    return start.elapsed_time(stop) / runs


def main():
    # 测试参数
    N1 = 400000
    N2 = 600000

    # 在 CPU 上创建随机数据
    a_cpu = torch.randn(N1, dtype=torch.float32)
    b_cpu = torch.randn(N2, dtype=torch.float32)

    # 将数据移到 GPU
    a, b = _check_inputs(a_cpu.cuda(), b_cpu.cuda())
    c = torch.empty((N1 + N2,), device=a.device, dtype=a.dtype)

    # 调用 Triton kernel
    launch_vector_concat(a, b, c)

    # CPU 计算结果作为对照
    c_ref = torch.cat([a_cpu, b_cpu], dim=0)

    # 验证结果
    if torch.allclose(c.cpu(), c_ref, atol=1e-5):
        print(f"测试通过! N1={N1}, N2={N2}")
        print(f"前5个结果: {c[:5].cpu().tolist()}")
    else:
        print("测试失败!")
        diff = (c.cpu() - c_ref).abs().max()
        print(f"最大误差: {diff}")
        return

    triton_ms = benchmark(lambda: launch_vector_concat(a, b, c))
    torch_ms = benchmark(lambda: torch.cat([a, b], dim=0))

    print(f"Triton kernel 平均耗时: {triton_ms:.6f} ms")
    print(f"torch.cat 平均耗时: {torch_ms:.6f} ms")


if __name__ == "__main__":
    main()