use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Cursor};
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs;
use std::path::Path;
use std::collections::HashSet;

// 引入新算法的结构体
use crate::ppann::Candidate;
use crate::pool::{*};
pub fn build_node_data(
    node_vector_arr: &[[f32; 128]], 
    node_neighbors: &[Vec<usize>]
) -> Vec<NodeData> {
    let mut node_data_list = Vec::with_capacity(node_vector_arr.len());

    for i in 0..node_vector_arr.len() {
        let mut node = NodeData::new_empty(); 
        node.vector.copy_from_slice(&node_vector_arr[i]);
        
        let current_neighbors = &node_neighbors[i];
        
        for (idx, &nid) in current_neighbors.iter().enumerate() {
            if idx < MAX_DEGREE { 
                node.neighbors[idx] = nid;
            } else {
                break;
            }
        }
        node_data_list.push(node);
    }
    
    node_data_list
}


pub fn read_nsg_graph(filename: &str) -> Vec<Vec<usize>> {
    let file = File::open(filename).expect("Failed to open file");
    let reader = BufReader::new(file);

    reader.lines()
        .skip(1) // 跳过第一行
        .filter_map(|line| line.ok().map(|line_content| {
            line_content
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect()
        }))
        .collect()
}

pub fn read_fvecs_file<const D: usize>(input_file_path: &str) -> io::Result<Vec<[f32; D]>> {
    let mut file = File::open(input_file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut result = Vec::new();
    let mut cursor = Cursor::new(buffer);

    while cursor.position() < cursor.get_ref().len() as u64 {
        let dimension = cursor.read_i32::<LittleEndian>()? as usize;
        assert_eq!(dimension, D, "Dimension mismatch!");

        let mut arr = [0.0; D];
        for j in 0..D {
            let value = cursor.read_f32::<LittleEndian>()?;
            arr[j] = value;
        }
        result.push(arr);
    }
    Ok(result)
}

pub fn read_certainty_node(file_path: &str) -> Option<Vec<usize>> {
    match fs::read_to_string(file_path) {
        Ok(content) => {
            let trimmed_content = content.trim();
            if trimmed_content.is_empty() {
                None
            } else {
                let ids: Vec<Result<usize, _>> = trimmed_content.split_whitespace().map(str::parse).collect();
                if ids.iter().any(|id| id.is_err()) {
                    None
                } else {
                    Some(ids.into_iter().map(|id| id.unwrap()).collect())
                }
            }
        }
        Err(_) => None, 
    }
}

pub fn load_ground_truth<P: AsRef<Path>>(filename: P) -> Vec<Vec<u32>> {
    let file = File::open(filename).unwrap();
    let mut reader = BufReader::new(file);

    let mut all_vectors = Vec::new();

    loop {
        let mut gk_bytes = [0u8; 4];
        if reader.read_exact(&mut gk_bytes).is_err() {
            break;
        }
        let gk = u32::from_le_bytes(gk_bytes);

        let mut vec = Vec::with_capacity(gk as usize);
        for _ in 0..gk {
            let mut elem_bytes = [0u8; 4];
            if reader.read_exact(&mut elem_bytes).is_err() {
                break;
            }
            vec.push(u32::from_le_bytes(elem_bytes));
        }
        all_vectors.push(vec);
    }

    all_vectors
}

// 适配新 Candidate 结构体的召回率计算
pub fn eval_recall(query_res: &Vec<Vec<Candidate>>, gts: &Vec<Vec<u32>>, k: usize) -> f32 {
    let mut mean_recall = 0.0;

    for (query, gt) in query_res.iter().zip(gts.iter()) {
        assert!(query.len() <= gt.len());
       
        let cur_query_res_set: HashSet<_> = query.iter().map(|item| item.id as u32).collect();
        let cur_query_gt: HashSet<_> = gt.iter().take(k).cloned().collect();

        let mut recall = 0.0;
        for x in &cur_query_res_set {
            if cur_query_gt.contains(&(*x)) {
                recall += 1.0;
            }
        }
        recall /= query.len() as f32;
        mean_recall += recall;
    }

    mean_recall /= query_res.len() as f32;
    mean_recall
}



pub fn read_initial_frequencies(file_path: &str, total_nodes: usize, y_queries: usize) -> Vec<f64> {
    let mut frequencies = vec![0.0f64; total_nodes];
    
    let file = match File::open(file_path) {
        Ok(f) => f,
        Err(e) => {
            println!("⚠️ 警告: 无法打开频率文件 '{}' ({})。将默认使用 0.0 初始化频率。", file_path, e);
            return frequencies;
        }
    };

    let reader = BufReader::new(file);
    let mut loaded_count = 0;

    for line in reader.lines() {
        if let Ok(line_content) = line {
            let parts: Vec<&str> = line_content.trim().split(',').collect();
            if parts.len() == 2 {
                if let (Ok(node_id), Ok(visit_count)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    if node_id < total_nodes {
                        // 核心计算逻辑： F[i] = 访问总数 / 5000
                        frequencies[node_id] = visit_count as f64 / y_queries as f64;
                        loaded_count += 1;
                    }
                }
            }
        }
    }
    
    println!("    [*] 成功从 {} 加载了 {} 个节点的初始访问频率", file_path, loaded_count);
    frequencies
}