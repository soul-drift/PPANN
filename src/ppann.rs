use core::time;
use std::cmp::Ordering;
use std::time::{Instant, Duration};
use crate::pool::{*};
use std::simd::f32x32;
use std::simd::num::SimdFloat;
use std::collections::BinaryHeap;
use std::cmp::Reverse;
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub id: usize,
    pub dist: f32,
    pub has_expanded: u32, 
}

impl Default for Candidate {
    fn default() -> Self {
        Candidate { id: usize::MAX, dist: f32::MAX, has_expanded: 1 }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool { self.dist.eq(&other.dist) }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub fn euclidean_distance(a: &[f32; 128], b: &[f32; 128]) -> f32 {
    let mut sum = 0.0;
    for i in 0..128 {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}
pub fn euclidean_distance_simd(a: &[f32; 128], b: &[f32; 128]) -> f32 {
    let mut sum = f32x32::splat(0.0);

    // 1. 处理可以被 32 整除的部分
    let chunk_size = 32;
    let limit = (128 / chunk_size) * chunk_size;

    for i in (0..limit).step_by(chunk_size) {
        let a_chunk = f32x32::from_slice(&a[i..i + chunk_size]);
        let b_chunk = f32x32::from_slice(&b[i..i + chunk_size]);
        let diff = a_chunk - b_chunk;
        sum += diff * diff;
    }

    // 把 SIMD 向量里的结果求和
    let mut total_sum = sum.reduce_sum();

    // 2. 处理尾部剩余元素
    for i in limit..128 {
        let diff = a[i] - b[i];
        total_sum += diff * diff;
    }

    total_sum
}
/// pp_ann: 搜索入口函数
/// 逻辑：从预置的 L3 Cache 中心节点（Hubs）中筛选出离查询向量 q 最近的一个，作为搜索起点。
pub fn pp_ann(
    pt: &mut UnifiedPool,           
    q: &[f32; 128],                 
    k: usize,     
    l: usize,                       
    t_0: usize,                     
    hub_nodes_in_l3: &[(usize, [f32; 128])], 
    expo:&mut[bool; 1000000]           
) -> Vec<Candidate> {
    
    let mut best_entry_id = usize::MAX;
    let mut min_dist = f32::MAX;
    // 遍历驻留在 L3 中的中心节点，利用 SIMD 快速定位最优入口点
    for (hub_id, hub_vec) in hub_nodes_in_l3.iter() {
        let dist = euclidean_distance_simd(hub_vec, q);
        if dist < min_dist {
            min_dist = dist;
            best_entry_id = *hub_id;
        }
    }
    // 进入隐私保护的图路由搜索
    obli_routing(pt, q, best_entry_id, min_dist, t_0, k, l,  expo)
}

/// 逻辑：结合 Oblivious RAM (ORAM) 思想，在前 15% 的步数维持强混淆，后 85% 转向极速读取。
pub fn obli_routing(
    pt: &mut UnifiedPool,           
    q: &[f32; 128],
    entry_id: usize,
    entry_dist: f32,
    t_0: usize,
    k: usize,
    l: usize,                     
    expo: &mut [bool; 1000000]           
) -> Vec<Candidate> {
    
    // 1. Min-Heap  (Frontier): 
    // 使用 Reverse 包裹 Candidate 使其成为最小堆。
    // 优先级规则：标记为 -1.0（待算距离）的节点永远排在最前面，其次是距离真实较近的节点。
    let mut min_heap: BinaryHeap<Reverse<Candidate>> = BinaryHeap::with_capacity(MAX_DEGREE * k);
    
    // 2. Max-Heap 结果池: 
    // 维护当前搜索到的全局最近 L 个节点，用于动态剪枝。
    let mut max_heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(l);
    // 压入起始节点
    let start_cand = Candidate {
        id: entry_id,
        dist: entry_dist, 
        has_expanded: 0, 
    };
    
    min_heap.push(Reverse(start_cand));
    max_heap.push(start_cand);
    expo[entry_id] = true;
    // 计算 15% 的阈值
    let phase1_limit = (t_0 as f64 * 0.15).ceil() as usize;
    // 启动固定次数 t_0 的搜索循环
    for step in 0..t_0 {
        // 从Min-Heap 提取优先级最高的节点
        let current_cand = match min_heap.pop() {
            Some(Reverse(cand)) => cand,
            None => break, // 如果Min-Heap清空，提前退出循环
        };

        let current_id = current_cand.id;
        let d_o = current_cand.dist;
        // 如果 dist != -1.0，说明这是一个已经计算过距离、现在准备扩展邻居的“扩展节点”。
        let need_expand_u32 = (d_o != -1.0) as u32;
        let need_expand_f32 = need_expand_u32 as f32;
        // 分阶段访问策略：
        // 前 15% 运行强混淆 pt.get（同时访问 Map 和普通节点并做无分支选择）。
        // 后 85% 运行 pt.get_normal_only（直接计算哈希位置读取）。
        let node_data = if step < phase1_limit {
            pt.get(current_id)
        } else {
            pt.get_normal_only(current_id)
        };
        // 计算当前节点到查询向量 q 的 SIMD 距离
        let actual_dist = euclidean_distance_simd(&node_data.vector, q);
        // 无分支状态流转逻辑：
        // 若当前节点是刚排队出来的（dist = -1.0），则 d_n 变为其实际距离。
        // 若当前节点是已计算过、正在扩展邻居的，则保留原距离。
        let d_n = (1.0 - need_expand_f32) * actual_dist + need_expand_f32 * d_o;
        // 状态机：如果刚刚完成了“排队计算距离”
        if need_expand_u32 == 0 {
            let mut updated_cand = current_cand;
            updated_cand.dist = d_n;
            updated_cand.has_expanded = 0;
            // 动态剪枝：只有当新计算的距离进入了全局 Top-L 范围，才允许将其放入探索队列
            if max_heap.len() < l || d_n < max_heap.peek().unwrap().dist {
                min_heap.push(Reverse(updated_cand)); 
                max_heap.push(updated_cand);          
                
                if max_heap.len() > l {
                    max_heap.pop(); // 踢出池子里离 q 最远的节点
                }
            }
        }
        // 标记已访问
        expo[current_id] = true;

        for &nid in node_data.neighbors.iter() {  
            // 当遇到 usize::MAX，说明真实的邻居已经遍历完了，直接 break
            if nid == usize::MAX {
                break;
            }
            
            // 只有当父节点已确定距离（need_expand=1）且邻居未曾露过面时，才准许入队
            let is_duplicate = expo[nid];
            let is_valid_push = need_expand_u32 == 1 && !is_duplicate;
            
            // 只有有效的节点，才压入队列中，并且马上标记为已发现
            if is_valid_push {
                expo[nid] = true; // 入队即标记：防止多个节点拥有相同邻居导致该邻居在 min_heap 中疯狂堆积
                
                let new_cand = Candidate {
                    id: nid,
                    dist: -1.0,  // // 置为 -1.0 标识，使其在 min_heap 中获得最高被弹出优先级
                    has_expanded: 0, 
                };
                min_heap.push(Reverse(new_cand));
            }
        }
    }
    // 从 max_heap 导出有序数组并截断到需要的 Top-K
    let mut final_results: Vec<Candidate> = max_heap.into_sorted_vec();
    final_results.retain(|node| node.dist >= 0.0 && node.dist < f32::MAX && node.id != usize::MAX);
    final_results.truncate(k);
    
    final_results
}

