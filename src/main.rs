#![feature(core_intrinsics)]
#![feature(portable_simd)]
use core::time;
use std::{fs::File, intrinsics, io::BufWriter, io::Write};
use std::time::Instant;

mod pool;
mod ppann;      
mod data_read;
mod sort;

use data_read::*;
use pool::*;
use ppann::{pp_ann, Candidate};

fn main() {
    // 1. 读取基础向量与 Query 数据
    let query_vector_arr =
        read_fvecs_file::<128>("/data/dbr/dbr_documents/sift/sift_query.fvecs").unwrap();
    let mut node_vector_arr =
        read_fvecs_file::<128>("/data/dbr/dbr_documents/sift/sift_base.fvecs").unwrap();
        
    // 2. 读取 NSG 图与中心节点
    let mut node_neighbors = read_nsg_graph("/data/dbr/dbr_documents/sift/nsg_sift.txt");
    let mut certainty_node =
        read_certainty_node("/data/dbr/dbr_documents/sift/sift_certainty_node_1.txt").unwrap();
    certainty_node.sort_unstable();
    
    // 3. 构建统一的 NodeData
    let node_data_list = build_node_data(&node_vector_arr, &node_neighbors);
    let total_normal_nodes = node_data_list.len();

    // 释放旧的内存占用
    node_vector_arr.clear();
    node_vector_arr.shrink_to_fit();
    node_neighbors.clear();
    node_neighbors.shrink_to_fit();

    // 4. 提取 L3 Cache 驻留中心节点
    let mut hub_nodes_in_l3 = Vec::new();
    for &hub_id in &certainty_node {
        hub_nodes_in_l3.push((hub_id, node_data_list[hub_id].vector));
    }

    // =====================================
    // 超参数配置
    // =====================================
    let t_0 = 2000;                 
    let k = 100;                    
    let l_search = 200;             
    
    // =====================================
    // 系统结构初始化 (不预分配任何 P_n 内存)
    // =====================================
    let mut pt = UnifiedPool::new(total_normal_nodes);

    // =====================================
    // 动态多 Epoch 循环运行架构
    // =====================================
    let y_queries_per_epoch = 5000;  // 频率数组分母数值
    let y_queries = 20_000;// 每次重构支撑 5000个查询
    let total_epochs = 1;        
    let mut query_batch_number = 0;

    // 【冷启动策略】：读取离线预热的真实频率文件
    let freq_file_path = "/home/ccy/ccy_documents/ppANN/ppann/data/sift_cold_node_visit_freq.txt"; 
    println!("--> 正在读取离线预热的真实频率分布...");
    let mut current_frequencies = read_initial_frequencies(
        freq_file_path, 
        total_normal_nodes, 
        y_queries_per_epoch
    );

    // 【核心改动】：将结果池移到最外层，用于收集所有 Epoch 的查询结果
    let mut all_query_results: Vec<Vec<Candidate>> = Vec::with_capacity(query_vector_arr.len());
    let total_search_start_time = Instant::now();

    for epoch in 1..=total_epochs {
        println!("\n========== 开始执行 Epoch {} ==========", epoch);

        // 1. 触发重建 
        pt.oblivious_reconstruct(
            &node_data_list, 
            &certainty_node,
            y_queries,
            &current_frequencies,
            query_batch_number
        );

        let epoch_search_start = Instant::now();

        // 截取属于当前 Epoch 的 Query 子集
        let start_idx = ((epoch - 1) * y_queries) % query_vector_arr.len();
        let current_query_batch = query_vector_arr.iter().skip(start_idx).take(y_queries);

        for query in current_query_batch {
            let mut expo: [bool; 1000000] = [false; 1000000];
            let result = pp_ann(
               &mut pt,
                query,
                k,
                l_search, 
                t_0,
                &hub_nodes_in_l3,
                &mut expo
            );
            // 收集当前 query 的结果到全局池
            all_query_results.push(result); 
        }
        
        println!("[*] Epoch {} 查询处理完毕，当前 Epoch 耗时: {:?}", epoch, epoch_search_start.elapsed());

        // 3. 导出真实访问频率分布，准备下一次进化重构
        println!("--> 保持当前epoch真实访问频率分布...");
        current_frequencies = pt.export_and_reset_frequencies(y_queries);
        query_batch_number += 1;
    }

    // =====================================
    // 最终阶段：统一验证全局召回率
    // =====================================
    println!("\n========== 所有查询处理完毕，开始验证召回率 ==========");
    println!("    [*] 全局总耗时: {:?}", total_search_start_time.elapsed());
    
    let gts = load_ground_truth("/data/dbr/dbr_documents/sift/sift_groundtruth.ivecs");
    
    // 确保用于对比的 ground truth 数量与实际收集到的查询结果数量一致
    let evaluated_gts: Vec<Vec<u32>> = gts.into_iter().take(all_query_results.len()).collect();
    
    let global_recall = eval_recall(&all_query_results, &evaluated_gts, k);
    println!("    [*] 最终全局召回率 (Global Recall@{}): {:.4}", k, global_recall);
}
