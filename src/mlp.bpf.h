#ifndef MLP_BPF_H
#define MLP_BPF_H

#include <vmlinux.h>
//返回a和b中的最大值
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
//返回a和b中的最小值
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// x: N
// W: MxN
//实现一个简单的全连接层，计算y = W * x，其中W是权重矩阵，x是输入向量，y是输出向量
static inline void linear_layer(const int32_t *W, const int32_t *x, int32_t *y, const int32_t M, const int32_t N)
{
    int64_t sum;//存储每个输出的加权和
    #pragma clang loop unroll(full) 
    //W是一个M * N的矩阵,对于每个输出元素 y[i]，计算该元素与输入向量 x 的加权和，即 y[i] = sum(W[i, :] * x)。
    for (int32_t i = 0; i < M; ++i) //遍历输出层的每一行（每个神经元）
    {
        sum = 0;//每个输出的加权和从0开始
        for (int32_t j = 0; j < N; ++j) //遍历输入向量的每一个元素
        {
            //将权重和输入值相乘，累加至sum
            sum += ((int64_t)(W[i * N + j])) * ((int64_t)(x[j]));
        }
        //输出值是sum的右16位，缩放，减少精度损失
        y[i] = sum >> 16;
    }
}
// ReLU 激活函数： y = max(x, 0)，将输入张量tensor中的每个元素小于0的值设置为0
static inline void relu(int32_t *tensor, const int32_t size)
{
    #pragma clang loop unroll(full)
    for (int32_t i = 0; i < size; i++)
        tensor[i] = MAX(tensor[i], 0);
}
// 标准化处理： y = (x - mean) / scale
//mean[i] 和 scale[i] 是每个输入维度的均值和标准差
// 用于将输入数据按均值和标准差进行标准化，常用于数据预处理
static inline void standard_scaler(int64_t *x, int32_t *y, int64_t *mean, int64_t *scale, int32_t N)
{
    #pragma clang loop unroll(full)
    for (int32_t i = 0; i < N; ++i)
    {
        if (mean[i] > x[i]) 
            y[i] = -((int32_t)(((uint64_t)mean[i] - (uint64_t)x[i]) * (1 << 16) / (uint64_t)scale[i]));
        else
            y[i] = (int32_t)(((uint64_t)x[i] - (uint64_t)mean[i]) * (1 << 16) / (uint64_t)scale[i]);
    }
}

#endif