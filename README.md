# PP-ANN: Privacy Preserving Approximate Nearest Neighbor

This repository contains a Rust implementation of **PP-ANN**, a privacy-preserving approximate nearest neighbor (ANN) search framework for graph-based vector retrieval. 

The current code is configured for **SIFT-style 128-dimensional vectors** and an NSG graph. It can be adapted to other vector dimensions and graph formats by changing the compile-time vector dimension and data paths.

---

## Repository Structure

```text
.
├── main.rs          
├── P_L_v2.rs        
├── ppann.rs
├── data_read.rs
└── sort.rs         # optional bitonic sort utilities
```

---

## Key Components and Functions

### `P_L_v2.rs`

#### `NodeData`

```rust
pub struct NodeData {
    pub vector: [f32; 128],
    pub neighbors: [usize; MAX_DEGREE],
}
```

Each graph node stores a fixed-size vector and up to `MAX_DEGREE` neighbor IDs. Empty neighbor slots are filled with `usize::MAX`.

#### `UnifiedPool`

```rust
pub struct UnifiedPool {
    pub hub_map: HashMap<usize, NodeData>,
    pub flat_pool: Vec<NodeData>,
    pub replica_limits: Vec<u32>,
    pub replica_prefix_sum: Vec<usize>,
    pub epoch_access_counts: Vec<u32>,
    ...
}
```

`UnifiedPool` separates the index into two logical regions:

- `P_h`: hub nodes stored in `hub_map`;
- `P_n`: normal nodes stored in `flat_pool`.

#### `UnifiedPool::oblivious_reconstruct`

It performs four steps:

1. **Hub pool construction**  
   Builds `P_h` from the selected hub nodes.

2. **Frequency-adaptive replica allocation**  
   Computes `replica_limits[i]` using the observed frequency `F[i]` and a Chernoff-style upper bound.

3. **Replica-position generation**  
   Generates the physical slot mapping `pos` using randomized active-node sampling and branch-minimized conditional swaps.

4. **Segmented parallel copy**  
   Copies vectors and neighbor lists into `flat_pool` through L3-cache-aware chunks and Rayon parallelism.

#### `UnifiedPool::perfect_hash`

Maps a logical access `(node_id, nth_replica)` to a physical memory slot. It uses a prefix-sum base offset plus a seeded LCG permutation. The parameter `a` is forced to be coprime with the pool size to avoid collisions in the physical address permutation.

#### `UnifiedPool::get`

Privacy-aware node access used in the early routing phase. It records the access count, performs both hub-side and normal-side access logic, and returns the correct result through a mask-based selection. If the allocated replica budget is exhausted, it performs a dummy read and returns an empty node to keep the access behavior regular.

#### `UnifiedPool::get_normal_only`

Fast normal-node access used in the later routing phase. It still consumes single-use replica slots and records access frequency, but removes the hub camouflage path to improve throughput.

#### `UnifiedPool::export_and_reset_frequencies`

Exports the empirical node access frequency of the current epoch and resets the counter. The returned distribution is used by the next reconstruction to adapt replica allocation.

---

### `ppann.rs`

#### `Candidate`

```rust
pub struct Candidate {
    pub id: usize,
    pub dist: f32,
    pub has_expanded: u32,
}
```

A candidate stores the node ID, distance to the query, and expansion state. Ordering is defined by distance so it can be used in Rust heaps.

#### `euclidean_distance_simd`

Computes squared Euclidean distance with `std::simd::f32x32`. This is used both for hub entry selection and graph traversal.

#### `pp_ann`

```rust
pub fn pp_ann(
    pt: &mut UnifiedPool,
    q: &[f32; 128],
    k: usize,
    l: usize,
    t_0: usize,
    hub_nodes_in_l3: &[(usize, [f32; 128])],
    expo: &mut [bool; 1000000],
) -> Vec<Candidate>
```

The top-level query interface. It first scans hub nodes to find the nearest entry point, then calls `obli_routing` for privacy-aware graph traversal.

#### `obli_routing`

The main graph search routine. It maintains:

- a min-heap frontier for candidate expansion;
- a max-heap result pool of size `L`;
- a visited bitmap `expo`;
- a fixed routing budget `t_0`.

The first `15%` of the routing budget uses `UnifiedPool::get`; the remaining `85%` uses `UnifiedPool::get_normal_only`. Nodes with unknown distances are inserted with `dist = -1.0`, then evaluated when popped from the frontier.

---

### `data_read.rs`

Key functions:

- `read_fvecs_file::<D>`: reads `.fvecs` vector files with dimension checking;
- `read_nsg_graph`: reads an NSG adjacency list;
- `read_certainty_node`: loads selected hub node IDs;
- `build_node_data`: combines vectors and NSG neighbors into `NodeData`;
- `read_initial_frequencies`: loads offline warm-up access frequencies;
- `load_ground_truth`: reads `.ivecs` ground-truth neighbors;
- `eval_recall`: computes mean Recall@K.

