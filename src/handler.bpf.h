// 处理网络流量的eBPF程序
#ifndef HANDLER_BPF_H
#define HANDLER_BPF_H

#include <vmlinux.h>
#include "common.h"
#include "mlp.bpf.h"
// 初始化flow_attribute结构体
static inline void init_flow_attribute(struct flow_attribute *attr)
{
    attr->num_packet = 0;
    attr->min_packet_length = 0x3fffffffffffffffULL;
    attr->max_duration = 0;
    attr->max_packet_length = 0;
    attr->dst_port = 0;
    attr->header_length = 0;

    attr->last_packet_time = 0;
    attr->status = 0;
}

static inline void update_flow_attribute(struct flow *f, struct tcphdr *tcp, u64 packet_length)
{
    // 获取当前数据包时间
    u64 packet_time = bpf_ktime_get_ns();
    // 记录特征提取的开始时间
    u64 extraction_start_time = bpf_ktime_get_ns();
    // 利用hash map存储，从map（flow_map)中查找f（数据包的四元组信息）
    struct flow_attribute *attr_ptr = (struct flow_attribute *)bpf_map_lookup_elem(&flow_map, f);
    // 未找到
    if (!attr_ptr)
    {
        // 初始化结构体attr
        struct flow_attribute attr = {};
        init_flow_attribute(&attr);
        // 更新为当前数据包时间戳
        attr.last_packet_time = packet_time;
        // 获取目的端口
        attr.dst_port = bpf_ntohs(tcp->dest);
        // 更新，将attr更新插入至flow_map中
        bpf_map_update_elem(&flow_map, f, &attr, BPF_NOEXIST);
        // 在flow_map中查找attr
        attr_ptr = (struct flow_attribute *)bpf_map_lookup_elem(&flow_map, f);
        // 未找到
        if (!attr_ptr)
            return;
    }
    // 找到数据包元组信息
    // 增加数据包数量
    attr_ptr->num_packet += 1;
    // 如果当前数据包长度小于流中记录的最小数据包长度
    if (packet_length < attr_ptr->min_packet_length)
        // 更新最小值
        attr_ptr->min_packet_length = packet_length;
    // 如果当前数据包长度大于流中记录的最小数据包长度
    if (packet_length > attr_ptr->max_packet_length)
    //更新最大值
        attr_ptr->max_packet_length = packet_length;
        //：当前数据包与上一个数据包之间的时间间隔
    u64 duration = packet_time - attr_ptr->last_packet_time;
    //当前时延大于之前记录的最大时延
    if (duration > attr_ptr->max_duration)
    //更新最大时延
        attr_ptr->max_duration = duration;
    //更新当前数据包时间
    attr_ptr->last_packet_time = packet_time;
    //更新TCP数据包头部长度：tcp->doff 表示 TCP 头部的长度（以 4 字节为单位）。乘以 4 后得到头部的字节数，并累加到流的总头部长度中。
    attr_ptr->header_length += tcp->doff * 4;
    //累加特征提取时间
    attr_ptr->total_feature_extraction_time += bpf_ktime_get_ns() - extraction_start_time;
    //根据TCP数据包的FIN/RST标志记录数据包结束
    if (tcp->fin || tcp->rst)
    {
        //记录流的检测开始时间
        attr_ptr->detection_start_time = bpf_ktime_get_ns();
    }
}

#endif