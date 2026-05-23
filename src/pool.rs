use std::collections::HashMap;
use xxhash_rust::xxh3::Xxh3;
use rand::Rng;
use rayon::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// ==========================================
// 全局配置与基础数据结构
// ==========================================

pub const MAX_DEGREE: usize = 50;

#[derive(Debug, Clone, Copy)]
pub struct NodeData {
    pub vector: [f32; 128], 
    pub neighbors: [usize; MAX_DEGREE], 
}

impl NodeData {
    pub fn new_empty() -> Self {
        NodeData {
            vector: [0.0; 128], 
            neighbors: [usize::MAX; MAX_DEGREE], 
        }
    }
}

// 确保完美哈希的互质性
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

// 无分支条件交换 (对齐伪代码 Line 20-21: ConditionalSwap)
#[inline(always)]
fn conditional_swap(arr: &mut [usize], i: usize, j: usize, condition: bool) {
    if i != j { 
        let mask = (0usize).wrapping_sub(condition as usize);
        let temp = (arr[i] ^ arr[j]) & mask;
        arr[i] ^= temp;
        arr[j] ^= temp;
    }
}

// ==========================================
// 核心结构：统一内存池 (UnifiedPool)
// ==========================================

pub struct UnifiedPool {
    pub hub_map: HashMap<usize, NodeData>,       // L3 Cache 中的 Hub 节点池 (P_h)
    pub flat_pool: Vec<NodeData>,                // DRAM 中的普通节点池 (P_n)
    pub total_normal_nodes: usize,               // 节点总数 |D|
    pub size_m: usize,                           // 动态物理容量总量 M = sum(A2[i])
    pub node_nth_table: Vec<u32>,                // 当前 Epoch 中各节点已使用的副本数
    pub hub_id_list: Vec<usize>,                 // Hub 节点 ID 列表
    pub param_a: usize,                          // LCG 参数 a
    pub param_b: usize,                          // LCG 参数 b
    
    pub replica_limits: Vec<u32>,                // 每个节点分配的动态副本数 A2[i]
    pub replica_prefix_sum: Vec<usize>,          // 前缀和，用于完美哈希的基址偏移
    pub epoch_access_counts: Vec<u32>,           // 记录每个节点在当前 Epoch 的真实访问量
}

impl UnifiedPool {

    /// 系统冷启动初始化 (不再需要预分配，起手为空)
    pub fn new(total_normal_nodes: usize) -> Self {
        UnifiedPool {
            hub_map: HashMap::new(),
            flat_pool: Vec::new(),  // 初始化为空，严格按照伪代码在运行中动态分配
            total_normal_nodes,
            size_m: 0,
            node_nth_table: vec![0; total_normal_nodes], 
            hub_id_list: Vec::new(),
            param_a: 1, 
            param_b: 0,
            replica_limits: vec![0; total_normal_nodes],
            replica_prefix_sum: vec![0; total_normal_nodes],
            epoch_access_counts: vec![0; total_normal_nodes],
        }
    }

    /// 对齐伪代码 Line 16: f(seed, id, c)
    #[inline(always)]
    fn perfect_hash(&self, node_id: usize, nth: usize) -> usize {
        let x_prime = self.replica_prefix_sum[node_id] + nth;
        (self.param_a.wrapping_mul(x_prime).wrapping_add(self.param_b)) % self.size_m
    }

    /// 导出当前 Epoch 的真实频率分布并清零
    pub fn export_and_reset_frequencies(&mut self, y_queries_executed: usize) -> Vec<f64> {
        let mut frequencies = Vec::with_capacity(self.total_normal_nodes);
        for i in 0..self.total_normal_nodes {
            let freq = self.epoch_access_counts[i] as f64 / y_queries_executed as f64;
            frequencies.push(freq);
        }
        self.epoch_access_counts.fill(0); 
        frequencies
    }

    /// =========================================================================
    /// 核心算法 2：Oblivious Single-Use Index 重建 (严格对齐版)
    /// =========================================================================
    pub fn oblivious_reconstruct(
        &mut self, 
        database: &[NodeData], 
        hub_nodes: &[usize], 
        y_queries: usize,              // 伪代码 Line 1 中的参数 y
        visit_frequencies: &[f64],     // 伪代码 Line 10 中的 F[i]
        new_batch_number: usize        // 伪代码 Line 1 中的 seed
    ) {
        let algo_start_time = std::time::Instant::now();
        
        // ---------------------------------------------------------
        // 伪代码 Lines 2-6: 构建 Hub Pool P_h
        // ---------------------------------------------------------
        let mut in_h = vec![0usize; self.total_normal_nodes];
        for &hid in hub_nodes { in_h[hid] = 1; }

        let mut p_h0 = vec![NodeData::new_empty(); hub_nodes.len()];
        let mut j = 0;
        
        for i in 0..self.total_normal_nodes {
            let safe_j = if j < p_h0.len() { j } else { 0 };
            p_h0[safe_j] = database[i]; 
            j += in_h[i]; // 伪代码 Line 6: ObliAppend           
        }

        self.hub_map.clear();
        self.hub_id_list.clear();
        for (idx, &hid) in hub_nodes.iter().enumerate() {
            self.hub_map.insert(hid, p_h0[idx]);
            self.hub_id_list.push(hid);
        }

        // ---------------------------------------------------------
        // 伪代码 Lines 7-10: 根据 Chernoff Bound 动态分配副本
        // ---------------------------------------------------------
        let c_const = 60.0 * std::f64::consts::LN_2; // 常数 C = 60 * ln(2)
        let mut current_prefix = 0;

        for i in 0..self.total_normal_nodes {
            let r_i = if in_h[i] == 1 { 0 } else {
                // 伪代码 Line 10: 解析解计算 A2[i] = ⌈(1 + \gamma_i) y F[i]⌉
                let mu = y_queries as f64 * visit_frequencies[i];
                let replicas_f64 = mu + (c_const + (c_const * c_const + 8.0 * c_const * mu).sqrt()) / 2.0;
                replicas_f64.ceil() as u32
            };
            
            self.replica_limits[i] = r_i;
            self.replica_prefix_sum[i] = current_prefix;
            current_prefix += r_i as usize;
        }
        
        self.size_m = current_prefix; 
        
        // 更新 LCG 参数
        let mut hasher = Xxh3::new();
        hasher.update(&new_batch_number.to_le_bytes());
        let hash_val = hasher.digest() as usize;
        let mut a = (hash_val >> 32) % self.size_m;
        let b = (hash_val & 0xFFFFFFFF) % self.size_m;
        if a == 0 { a = 1; }
        while gcd(a, self.size_m) != 1 {
            a = (a + 1) % self.size_m;
            if a == 0 { a = 1; } 
        }
        self.param_a = a;
        self.param_b = b;
        
        // 伪代码 Line 28: initialize cnt[id] = 0 (用 node_nth_table 代表 cnt)
        self.node_nth_table.fill(0); 

        // 伪代码 Line 8: initialize arrays A1 and A2
        let mut a1 = Vec::with_capacity(self.total_normal_nodes);
        let mut a2 = Vec::with_capacity(self.total_normal_nodes);
        for i in 0..self.total_normal_nodes {
            if in_h[i] == 0 {
                a1.push(i);
                a2.push(self.replica_limits[i] as usize);
            }
        }
        
        // ---------------------------------------------------------
        // 伪代码 Lines 11-22: Compute the replica-position array pos
        // ---------------------------------------------------------
        // 伪代码 Line 11: initialize pos of size \sum_i A_2[i]
        let mut pos: Vec<usize> = vec![0; self.size_m];
        let mut active = a1.len(); 
        let mut rng = SmallRng::from_entropy();

        while active > 0 {
            let tail = active - 1;
            let idx = rng.gen_range(0..=tail);
            
            let id = a1[idx];                 
            let remaining_c = a2[idx];        
            
            let nth = self.replica_limits[id] as usize - remaining_c;
            let mapped_idx = self.perfect_hash(id, nth); 
            pos[mapped_idx] = id;                        
            
            a2[idx] -= 1;                                
            let is_full = a2[idx] == 0;                  
            
            conditional_swap(&mut a1, idx, tail, is_full); 
            conditional_swap(&mut a2, idx, tail, is_full); 
            active -= is_full as usize;                    
        }

        // ---------------------------------------------------------
        // 伪代码 Line 23: initialize P_n with |pos| fixed-size slots
        // ---------------------------------------------------------
        println!("    [*] 正在精确申请 {} 个节点的物理内存...", self.size_m);
        let alloc_start = std::time::Instant::now();
        
        // 💡 严格遵循伪代码，在此处精确分配所需的物理内存。
        self.flat_pool = vec![NodeData::new_empty(); self.size_m];
        
        // 💡 单独记录操作系统分配和清零这块内存耗费的时间
        let alloc_duration = alloc_start.elapsed();
        println!("    [*] 操作系统物理内存分配完成，耗时: {:?}", alloc_duration);

        // ---------------------------------------------------------
        // 伪代码 Lines 24-27: segment copy
        // ---------------------------------------------------------
        let dim = database.first().map_or(960, |n| n.vector.len());
        let max_deg = MAX_DEGREE;
        let safe_l3_cache_bytes = 96 * 1024 * 1024; // 80MB
        
        let vec_chunk_size = std::cmp::max(1, safe_l3_cache_bytes / (self.total_normal_nodes * 4));
        let vec_passes = (dim + vec_chunk_size - 1) / vec_chunk_size; 

        for pass in 0..vec_passes {
            let start_dim = pass * vec_chunk_size;
            let end_dim = std::cmp::min(start_dim + vec_chunk_size, dim);
            let actual_chunk_size = end_dim - start_dim;
            
            let mut a3 = vec![0.0f32; self.total_normal_nodes * actual_chunk_size];
            for i in 0..self.total_normal_nodes {
                a3[i * actual_chunk_size .. (i + 1) * actual_chunk_size]
                    .copy_from_slice(&database[i].vector[start_dim..end_dim]);
            }
            
            self.flat_pool.par_iter_mut().zip(pos.par_iter()).for_each(|(pool_node, &src_node_id)| {
                pool_node.vector[start_dim..end_dim]
                    .copy_from_slice(&a3[src_node_id * actual_chunk_size .. (src_node_id + 1) * actual_chunk_size]);
            });
        }

        let neighbor_chunk_size = std::cmp::max(1, safe_l3_cache_bytes / (self.total_normal_nodes * 8));
        let neighbor_passes = (max_deg + neighbor_chunk_size - 1) / neighbor_chunk_size;

        for pass in 0..neighbor_passes {
            let start_dim = pass * neighbor_chunk_size;
            let end_dim = std::cmp::min(start_dim + neighbor_chunk_size, max_deg);
            let actual_chunk_size = end_dim - start_dim;
            
            let mut a3_neighbors = vec![0usize; self.total_normal_nodes * actual_chunk_size];
            for i in 0..self.total_normal_nodes {
                a3_neighbors[i * actual_chunk_size .. (i + 1) * actual_chunk_size]
                    .copy_from_slice(&database[i].neighbors[start_dim..end_dim]);
            }
            
            self.flat_pool.par_iter_mut().zip(pos.par_iter()).for_each(|(pool_node, &src_node_id)| {
                pool_node.neighbors[start_dim..end_dim]
                    .copy_from_slice(&a3_neighbors[src_node_id * actual_chunk_size .. (src_node_id + 1) * actual_chunk_size]);
            });
        }
        
        let total_duration = algo_start_time.elapsed();
        // 💡 算法开销 = 总耗时 - 操作系统物理内存申请开销
        let pure_algo_duration = total_duration.saturating_sub(alloc_duration);
        
        println!("==> Algorithm 2 重建完成！");
        println!("    --> 真实经过时间: {:?}", total_duration);
        println!("    --> 纯算法执行耗时 (已扣除OS分配): {:?}", pure_algo_duration);
    }

    // ==========================================
    // 搜索期获取函数
    // ==========================================

    pub fn get(&mut self, target_id: usize) -> NodeData {
        if target_id < self.total_normal_nodes {
            self.epoch_access_counts[target_id] = self.epoch_access_counts[target_id].saturating_add(1);
        }

        let mut rng = rand::thread_rng();
        let is_hub_mask = self.hub_map.contains_key(&target_id) as usize; 

        let random_normal_id = rng.gen_range(0..self.total_normal_nodes); 
        let random_hub_id = if !self.hub_id_list.is_empty() {
            let random_hub_idx = rng.gen_range(0..self.hub_id_list.len());
            self.hub_id_list[random_hub_idx]
        } else {
            0
        };

        let actual_hub_target = is_hub_mask * target_id + (1 - is_hub_mask) * random_hub_id;
        let actual_hash_target = is_hub_mask * random_normal_id + (1 - is_hub_mask) * target_id;

        let empty_hub_data = NodeData::new_empty();
        let hub_data_candidate = self.hub_map.get(&actual_hub_target).unwrap_or(&empty_hub_data);

        let table_len = self.node_nth_table.len();
        let nth_index = actual_hash_target % table_len;
        let current_nth = self.node_nth_table[nth_index];
        self.node_nth_table[nth_index] = self.node_nth_table[nth_index].saturating_add(1);

        let normal_data_candidate = if current_nth >= self.replica_limits[nth_index] {
            let dummy_index = self.perfect_hash(0, 0); 
            let _dummy_data = self.flat_pool[dummy_index]; 
            NodeData::new_empty()
        } else {
            let index = self.perfect_hash(actual_hash_target, current_nth as usize); 
            self.flat_pool[index]
        };

        let final_choices = [&normal_data_candidate, hub_data_candidate];
        *final_choices[is_hub_mask]
    }

    pub fn get_normal_only(&mut self, target_id: usize) -> NodeData {
        if target_id < self.total_normal_nodes {
            self.epoch_access_counts[target_id] = self.epoch_access_counts[target_id].saturating_add(1);
        }

        let table_len = self.node_nth_table.len();
        let nth_index = target_id % table_len;
        
        let current_nth = self.node_nth_table[nth_index];
        self.node_nth_table[nth_index] = self.node_nth_table[nth_index].saturating_add(1);

        if current_nth >= self.replica_limits[nth_index] {
            let dummy_index = self.perfect_hash(0, 0); 
            let _dummy_data = self.flat_pool[dummy_index]; 
            return NodeData::new_empty();
        }

        let index = self.perfect_hash(target_id, current_nth as usize); 
        self.flat_pool[index]
    }
}