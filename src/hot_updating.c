// eBPF更新神经网络参数并修改神经网络索引
#include <bpf/bpf.h>
#include <stdio.h>
#include "mlp_params.bpf.h"

// 神经网络参数结构体
struct Net
{
    // 3层权重数组
    int32_t layer_0_weight[192];
    int32_t layer_1_weight[1024];
    int32_t layer_2_weight[64];
    int64_t mean[6];
    int64_t scale[6];
};

int main()
{
    // 神经网络参数存储位置BPF文件系统
    const char *nn_parameters_path = "/sys/fs/bpf/nn_parameters";
    // 神经网络索引的存储路径BPF文件系统
    const char *nn_idx_path = "/sys/fs/bpf/nn_idx";
    // 获取BPF对象文件描述符
    int nn_parameters_fd = bpf_obj_get(nn_parameters_path);
    int nn_idx_fd = bpf_obj_get(nn_idx_path);
    // 键
    int zero_idx = 0;
    // 神经网络索引值0 or 1
    int nn_idx = 0;
    // 从nn_idx_fd中查找nn_idx
    int err = bpf_map_lookup_elem(nn_idx_fd, &zero_idx, &nn_idx);
    printf("load nn index error: %s\n", strerror(err));
    printf("NN index: %d\n", nn_idx);
   
    /*
    切换神经网络索引 ，热更新机制，
    假设有两个神经网络参数集合，一个正在使用，另一个准备好供下一个周期使用。
    这个机制可以用来进行 双缓冲，即一边使用一组神经网络参数，另一边准备更新后的参数。
    通过轮换索引，可以保证每次使用的神经网络都有最新的参数。

    详细解释：
    nn_idx：当前神经网络的索引，可能是 0 或 1，表示当前使用的神经网络的参数。
    (nn_idx + 1) % 2：通过对 nn_idx + 1 取模 2 来切换索引，确保值始终在 0 和 1 之间。
    如果 nn_idx 为 0，则 (0 + 1) % 2 结果为 1。
    如果 nn_idx 为 1，则 (1 + 1) % 2 结果为 0。
    */

    int new_nn_idx = (nn_idx + 1) % 2;
    struct Net net;

    // err = bpf_map_lookup_elem(nn_parameters_fd, &new_nn_idx, &net);
    // printf("load ready nn error: %s\n", strerror(err));
    // 将神经网络的参数（mean、scale、权重）从预定义的数组复制到 net 结构体中的相应字段
    // mean
    memcpy(net.mean, mean, sizeof(int64_t) * N0);
    // scale
    memcpy(net.scale, scale, sizeof(int64_t) * N0);
    // layer 0
    memcpy(net.layer_0_weight, layer_0_weight, sizeof(int32_t) * N1 * N0);
    // layer 1
    memcpy(net.layer_1_weight, layer_1_weight, sizeof(int32_t) * N2 * N1);
    // layer 2
    memcpy(net.layer_2_weight, layer_2_weight, sizeof(int32_t) * N3 * N2);

    printf("new NN index: %d\n", new_nn_idx);
    // 更新new_nn_idx索引对应的神经网络参数，将net保存到nn_parameters_fd，key：new_nn_idx
    err = bpf_map_update_elem(nn_parameters_fd, &new_nn_idx, &net, BPF_ANY);
    printf("update nn error: %s\n", strerror(err));
    // 更新神经网络索引，将new_nn_idx更新至nn_idx_fd，key:zero_idx
    err = bpf_map_update_elem(nn_idx_fd, &zero_idx, &new_nn_idx, BPF_ANY);
    printf("update nn idx error: %s\n", strerror(err));
}