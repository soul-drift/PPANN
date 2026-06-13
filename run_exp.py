import argparse
import re
import os
import subprocess
import sys

def update_file(filepath, pattern_replacements):
    """读取文件，应用正则替换，有变化则覆写"""
    if not os.path.exists(filepath):
        print(f"⚠️ 找不到文件: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for pattern, repl in pattern_replacements:
        # 使用正则表达式替换匹配到的部分
        new_content = re.sub(pattern, repl, new_content)

    if content != new_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ 已更新文件: {filepath}")
    else:
        print(f"❄️ 文件无需修改: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="动态修改 Rust PP-ANN 参数并自动编译运行 (动态频率版)")
    
    # 文件路径参数
    parser.add_argument('--query_path', type=str, required=True, help='Query数据集(fvecs)路径')
    parser.add_argument('--base_path', type=str, required=True, help='Base向量集(fvecs)路径')
    parser.add_argument('--graph_path', type=str, required=True, help='NSG图文件(txt)路径')
    parser.add_argument('--certainty_path', type=str, required=True, help='中心节点文件(txt)路径')
    parser.add_argument('--gt_path', type=str, required=True, help='Ground Truth文件(ivecs)路径')
    # 【新增】离线预热频率文件路径
    parser.add_argument('--freq_path', type=str, required=True, help='离线节点访问频率文件(txt)路径')

    # 模型与系统超参数
    parser.add_argument('--dim', type=int, default=128, help='向量维度 (例如 SIFT为128, GIST为960)')
    parser.add_argument('--total_nodes', type=int, default=1000000, help='基础向量总数 (决定防重数组大小)')
    parser.add_argument('--max_degree', type=int, default=50, help='图的最大邻居数 MAX_DEGREE')
    parser.add_argument('--k', type=int, default=100, help='返回的 Top-k 数量')
    parser.add_argument('--l_search', type=int, default=200, help='搜索队列长度 l_search')
    parser.add_argument('--t_0', type=int, default=6000, help='每次查询的固定跳数 t_0')

    args = parser.parse_args()

    # 注意：确保这里的文件路径匹配你实际的 src/ 目录结构
    
    # 1. 修改 src/pool.rs
    update_file('src/pool.rs', [
        (r'pub const MAX_DEGREE: usize = \d+;', f'pub const MAX_DEGREE: usize = {args.max_degree};'),
        (r'\[f32; \d+\]', f'[f32; {args.dim}]'),
        (r'\[0\.0; \d+\]', f'[0.0; {args.dim}]'),
    ])

    # 2. 修改 src/ppann.rs
    # 2. 修改 src/ppann.rs
    update_file('src/ppann.rs', [
        (r'\[f32; \d+\]', f'[f32; {args.dim}]'),
        (r'\(0\.\.\d+\)\.step_by\(32\)', f'(0..{args.dim}).step_by(32)'),
        (r'\[bool; \d+\]', f'[bool; {args.total_nodes}]'),
        (r'let limit = .*?;', f'let limit = ({args.dim} / chunk_size) * chunk_size;'),
        (r'(?<=limit\.\.)\d+', f'{args.dim}'),
        (r'(?<=\b0\.\.)\d+', f'{args.dim}')
    ])

    # 3. 修改 src/data_read.rs
    update_file('src/data_read.rs', [
        (r'\[f32; \d+\]', f'[f32; {args.dim}]'),
    ])

    # 4. 修改 src/main.rs
    update_file('src/main.rs', [
        # 替换读文件路径与类型泛型
        (r'(query_vector_arr\s*=\s*read_fvecs_file::<)\d+(>\()".*?"(\))', rf'\g<1>{args.dim}\g<2>"{args.query_path}"\g<3>'),
        (r'(node_vector_arr\s*=\s*read_fvecs_file::<)\d+(>\()".*?"(\))', rf'\g<1>{args.dim}\g<2>"{args.base_path}"\g<3>'),
        (r'(node_neighbors\s*=\s*read_nsg_graph\()".*?"(\))', rf'\g<1>"{args.graph_path}"\g<2>'),
        (r'(certainty_node\s*=\s*read_certainty_node\()".*?"(\))', rf'\g<1>"{args.certainty_path}"\g<2>'),
        (r'(gts\s*=\s*load_ground_truth\()".*?"(\))', rf'\g<1>"{args.gt_path}"\g<2>'),
        
        # 【新增】替换预热频率文件的路径
        (r'(let freq_file_path\s*=\s*)".*?"(;)', rf'\g<1>"{args.freq_path}"\g<2>'),

        # 替换超参数 (适配小写的变量名)
        (r'let t_0\s*=\s*\d+;', f'let t_0 = {args.t_0};'),
        (r'let k\s*=\s*\d+;', f'let k = {args.k};'),
        (r'let l_search\s*=\s*\d+;', f'let l_search = {args.l_search};'),

        # 替换 bool 查重数组的写死大小
        (r'\[bool; \d+\]', f'[bool; {args.total_nodes}]'),
        (r'\[false; \d+\]', f'[false; {args.total_nodes}]'),
    ])

    # 5. 执行 Cargo 编译与运行
    print("\n🚀 代码参数替换完毕，正在唤起 Rust 编译器...")
    cmd = 'RUSTFLAGS="-C target-cpu=native" cargo run --release'
    
    # 实时透传终端输出
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    
    if process.returncode != 0:
        print("\n❌ 运行失败，请检查上方编译报错信息。")
        sys.exit(process.returncode)
    else:
        print("\n🎉 实验运行结束！")

if __name__ == '__main__':
    main()
