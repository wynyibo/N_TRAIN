#ifndef PARAMS_BPF_H
#define PARAMS_BPF_H
//利用eBPF Map存储神经网络参数等
struct Net
{
	int32_t layer_0_weight[192]; //输入层-隐藏层权重 192个
	int32_t layer_1_weight[1024];//第一隐藏层-第二隐藏层权重 1024个
	int32_t layer_2_weight[64]; //第二隐藏层-输出层权重 64个
    int64_t mean[6]; //输入数据的均值，用于归一化处理
    int64_t scale[6]; //输入数据缩放因子，用于归一化处理
};

/*
在bpf_heplers.h中，枚举定义如下：

enum libbpf_pin_type {
    LIBBPF_PIN_NONE,         // 不固定映射
    LIBBPF_PIN_BY_NAME,      // 按名称固定映射
};

针对LIBBPF_PIN_NONE类型，程序退出/内核重新加载时映射会丢失，适用场景:临时/不需长期存在的映射
针对LIBBPF_PIN_BY_NAME类型，固定映射位置，持久存放于指定文件路径下，通常是/sys/fs/bpf/ ，适用场景：长期存储的映射/长期保留

*/
struct
{
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__type(key, int32_t);
	__type(value, struct Net);
	__uint(max_entries, 2);
	__uint(pinning, LIBBPF_PIN_BY_NAME); // <- pin   
} nn_parameters SEC(".maps");

struct
{
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__type(key, int32_t);
	__type(value, int32_t);
	__uint(max_entries, 1);
	__uint(pinning, LIBBPF_PIN_BY_NAME); // <- pin
} nn_idx SEC(".maps");

#endif