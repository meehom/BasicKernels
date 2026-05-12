#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// 两个输入向量的大小
#define N1 400000
#define N2 600000

// CUDA内核函数 - 执行向量拼接
__global__ void vectorConcat(float *a, float *b, float *c, int n1, int n2)
{
    // 计算当前线程处理的元素索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int total = n1 + n2;

    // 确保线程不会越界访问数组
    if (i < total)
    {
        if (i < n1)
        {
            c[i] = a[i];
        }
        else
        {
            c[i] = b[i - n1];
        }
    }
}

int main()
{
    // 主机内存指针
    float *h_a, *h_b, *h_c;
    // 设备内存指针
    float *d_a, *d_b, *d_c;

    size_t size_a = N1 * sizeof(float);
    size_t size_b = N2 * sizeof(float);
    size_t size_c = (N1 + N2) * sizeof(float);

    // 分配主机内存
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);

    // 初始化输入向量
    for (int i = 0; i < N1; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < N2; i++)
    {
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // 分配设备内存
    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c);

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // 计算线程块和网格大小
    int total = N1 + N2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    // 启动内核
    vectorConcat<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N1, N2);

    // 检查内核启动是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < N1; i++)
    {
        if (fabs(h_a[i] - h_c[i]) > 1e-5)
        {
            printf("Result verification failed at element %d!\n", i);
            break;
        }
    }
    for (int i = 0; i < N2; i++)
    {
        if (fabs(h_b[i] - h_c[N1 + i]) > 1e-5)
        {
            printf("Result verification failed at element %d!\n", N1 + i);
            break;
        }
    }

    printf("Vector concatenation completed successfully!\n");

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 释放主机内存
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}