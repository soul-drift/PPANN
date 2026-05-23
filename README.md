# PPANN: Privacy Preserving Approximate Nearest Neighbor

This repository contains a Rust implementation of **PPANN**, a privacy-preserving approximate nearest neighbor (ANN) search framework in TEEs.

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

## Build and Run

This project uses unstable Rust features:

```rust
#![feature(core_intrinsics)]
#![feature(portable_simd)]
```

Use a nightly Rust toolchain:

```bash
rustup default nightly
cargo run --release
```

Required crates include:

```toml
[dependencies]
byteorder = "1"
rand = { version = "0.8", features = ["small_rng"] }
rayon = "1"
xxhash-rust = { version = "0.8", features = ["xxh3"] }
```

Before running, make sure that the dataset paths in `main.rs` are changed to your local environment.

Or you can use Python scripts  `run_exp.py`  for automated PP-ANN experiments. It dynamically modifies file paths and system hyperparameters within the Rust source code based on the provided command-line arguments, and automatically triggers `cargo run --release` to compile and execute. This eliminates the tedious process of manually editing the source code every time you change datasets or parameters.

```
$ cd PPANN/
$ python run_exp.py \
   --query_path QUERY_PATH \
   --base_path BASE_PATH \
   --graph_path GRAPH_PATH \
   --certainty_path CERTAINTY_PATH \
   --gt_path GT_PATH \
   --freq_path FREQ_PATH \
   --dim DIM \
   --total_nodes TOTAL_NODES \
   --max_degree MAX_DEGREE \
   --k K \
   --l_search L_SEARCH \
   --t_0 T_0
```

## Detailed Parameter Description

The arguments used in the command above cover two main categories: **File Paths** and **System Hyperparameters**.

### File Path Parameters

- **`--query_path`**: Query dataset path. Specifies the query vector file in `fvecs` format (e.g., `sift_query.fvecs`).
- **`--base_path`**: Base vector set path. Specifies the base gallery vector file in `fvecs` format (e.g., `sift_base.fvecs`).
- **`--graph_path`**: NSG graph file path. Specifies the text file containing the pre-built NSG graph structure(e.g., `nsg_sift.txt`).
- **`--certainty_path`**: Hub node file path. Specifies the text file containing the selected hub (certainty) nodes.
- **`--gt_path`**: Ground Truth file path. Specifies the ground truth file in `ivecs` format, used for evaluating the final Recall.
- **`--freq_path`**: Offline node access frequency file path. Specifies the text file containing warm-up frequency data, used for the frequency-adaptive replica allocation mechanism.

### Model & System Hyperparameters

- **`--dim`**: Vector dimension. For example, SIFT is typically `128`, and GIST is `960`. The script automatically updates all array length definitions in the Rust code based on this parameter.
- **`--total_nodes`**: Total number of base vectors. For example, `1000000` for SIFT1M. This parameter determines the memory allocation size for the deduplication array (visited bitmap).
- **`--max_degree`**: Maximum neighbor limit of the graph. Determines the `MAX_DEGREE` constant size in the NodeData struct. Configured to `50` in the example.
- **`--k`**: Target search result count. The number of Top-K results to return per query. Configured to `100` in the example.
- **`--l_search`**: Candidate search queue length. The size of the candidate pool maintained during graph traversal. Configured to `200` in the example.
- **`--t_0`**: Fixed search steps (hops). The fixed routing budget used to mask data-dependent memory access patterns. It controls the trade-off between search time and privacy strength. Configured to `2000` in the example.

### Notes

1. **Code Modification Backup**: Because this script uses regular expressions to **directly overwrite** the Rust source files in the `src/` directory, it is highly recommended to commit your code to a Git repository before use to prevent unexpected replacements from corrupting your code.
2. **Compilation Environment**: The script will automatically invoke `cargo run --release` with the `-C target-cpu=native` flag enabled for compilation optimization. Please ensure that a Nightly Rust toolchain with SIMD support is correctly installed in your running environment.



---

## Key Components and Functions

### `P_L_v2.rs`

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

