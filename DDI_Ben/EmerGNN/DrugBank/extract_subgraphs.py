"""
extract_subgraphs.py — Trích xuất TIGHT enclosing subgraph cho EmerGNN.

Công thức:  V_tight(h, t, L) = { j : d(h,j) + d(j,t) ≤ L }

Đảm bảo hiddens_L[t] trên sub-KG GIỐNG HỆT chạy trên full KG.

Usage:
    # Extract cho tất cả settings + splits (mặc định L=3)
    python extract_subgraphs.py

    # Chọn scenario + split cụ thể, float16
    python extract_subgraphs.py --scenarios S1 --splits test --dtype float16

    # L=2, compact hơn
    python extract_subgraphs.py --L 2

Output:
    data/subgraphs/S0_train_L3.npz
    data/subgraphs/S0_valid_L3.npz
    data/subgraphs/S0_test_L3.npz
    ... (mỗi scenario × split 1 file)

Mỗi .npz chứa:
    offsets      : int32[N_pairs+1]  — ragged array offset cho các node, subgraph_i = nodes[offsets[i]:offsets[i+1]]
    nodes        : uint16[]          — các ID node được nối tiếp nhau (concatenated)
    heads        : uint16[N_pairs]   — ID thuốc đầu (head drug)
    tails        : uint16[N_pairs]   — ID thuốc cuối (tail drug)
    rels         : uint8[N_pairs]    — loại quan hệ DDI cần dự đoán (< 86)
    n_nodes      : uint16[N_pairs]   — số lượng node trong subgraph
    edge_offsets : int32[N_pairs+1]  — ragged array offset cho các cạnh, edges_i = edge_heads/tails[edge_offsets[i]:edge_offsets[i+1]]
    edge_heads   : uint16[]          — danh sách nút đầu của các cạnh thực tế
    edge_tails   : uint16[]          — danh sách nút cuối của các cạnh thực tế
    edge_rels    : uint8[]           — danh sách quan hệ thực tế tương ứng với các cạnh
    meta         : dict              — {L, scenario, split, n_entities, ...}
"""

import os
import argparse
import json
import time
import numpy as np
from scipy import sparse
from tqdm import tqdm


# ────────────────────── Config ──────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ALL_SCENARIOS = ['S0', 'S1', 'S2']
ALL_SPLITS = ['train', 'valid', 'test']


# ────────────────────── Data loading ──────────────────────

def load_triplets(path):
    """Load (h, t, r) triplets từ text file."""
    triplets = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triplets.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return np.array(triplets, dtype=np.int32) if triplets else np.empty((0, 3), dtype=np.int32)


def build_undirected_csr(triplets_list, n_nodes):
    """Build undirected CSR adjacency (binary, deduplicated)."""
    rows_all, cols_all = [], []
    for triplets in triplets_list:
        if len(triplets) == 0:
            continue
        h = triplets[:, 0].astype(np.int64)
        t = triplets[:, 1].astype(np.int64)
        mask = h != t
        rows_all.extend([h[mask], t[mask]])
        cols_all.extend([t[mask], h[mask]])
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    sp = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_nodes, n_nodes),
    )
    sp.sum_duplicates()
    sp.data[:] = 1
    return sp


def load_all_data():
    """Load KG + DDI splits + compute n_entities."""
    kg = load_triplets(os.path.join(DATA_DIR, 'KG.txt'))
    with open(os.path.join(DATA_DIR, 'node2id.json')) as f:
        node2id = json.load(f)
    with open(os.path.join(DATA_DIR, 'entity_drug.json')) as f:
        entity2id = json.load(f)

    n_drugs = len(node2id)
    n_entities = max(max(entity2id.values()), max(int(v) for v in node2id.values())) + 1
    n_entities = max(n_entities, int(kg[:, :2].max()) + 1)

    ddi = {}
    for sc in ALL_SCENARIOS:
        ddi[sc] = {}
        for sp in ALL_SPLITS:
            path = os.path.join(DATA_DIR, sc, f'{sp}_ddi.txt')
            ddi[sc][sp] = load_triplets(path)

    return kg, ddi, n_drugs, n_entities


def build_graph_for_split(kg, ddi, scenario, split):
    """Build đúng graph mà EmerGNN dùng cho từng split (theo load_data.py).

    train → KG + full train_ddi  (superset — bao gồm cả fact+train_data)
    valid → KG + train_ddi       (= vKG)
    test  → KG + train_ddi + valid_ddi  (= tKG)
    """
    triplet_sets = [kg, ddi[scenario]['train']]
    if split in ('valid', 'test'):
        pass  # train_ddi đã include
    if split == 'test':
        triplet_sets.append(ddi[scenario]['valid'])
    return triplet_sets


# ────────────────────── BFS ──────────────────────

def bfs_distances(sp_csr, source, exclude_node, max_depth):
    """BFS trả về int8 distance array. Unreachable = max_depth+1.
    Loại bỏ cạnh trực tiếp nối giữa source và exclude_node ở bước đầu tiên.
    """
    n = sp_csr.shape[0]
    UNREACH = max_depth + 1
    dist = np.full(n, UNREACH, dtype=np.int8)
    dist[source] = 0
    frontier = np.array([source], dtype=np.int64)
    for d in range(1, max_depth + 1):
        if len(frontier) == 0:
            break
        # Lấy tất cả neighbor của frontier nodes qua CSR indexing
        nbrs = sp_csr[frontier].indices
        if len(nbrs) == 0:
            break
        if d == 1:
            nbrs = nbrs[nbrs != exclude_node]
        new_mask = dist[nbrs] > d
        new_nodes = np.unique(nbrs[new_mask])
        if len(new_nodes) == 0:
            break
        dist[new_nodes] = d
        frontier = new_nodes
    return dist


def extract_tight_subgraph(sp_csr, h, t, L):
    """Trả về sorted array of node IDs in V_tight(h, t, L).

    V_tight = { j : d(h,j) + d(j,t) ≤ L }

    Nếu d(h,t) > L → return empty array (cặp unreachable).
    """
    d_h = bfs_distances(sp_csr, h, t, L)
    d_t = bfs_distances(sp_csr, t, h, L)
    # int8 + int8 → int16 (tránh overflow)
    tight_mask = (d_h.astype(np.int16) + d_t.astype(np.int16)) <= L
    nodes = np.where(tight_mask)[0]
    return nodes  # đã sorted (np.where trả sorted)


# ────────────────────── Main extraction ──────────────────────

def extract_scenario_split(kg, ddi, scenario, split, L, n_entities, out_dir, dtype):
    """Extract + save tight enclosing subgraphs cho 1 (scenario, split)."""

    pairs = ddi[scenario][split]
    n_pairs = len(pairs)
    if n_pairs == 0:
        print(f'  {scenario}/{split}: 0 pairs → skip')
        return

    # Build graph
    triplet_sets = build_graph_for_split(kg, ddi, scenario, split)
    adj = build_undirected_csr(triplet_sets, n_entities)

    # Build global adjacency list for extracting induced subgraph edges
    adj_list = [[] for _ in range(n_entities)]
    for triplets in triplet_sets:
        for h_edge, t_edge, r_edge in triplets:
            h_edge, t_edge, r_edge = int(h_edge), int(t_edge), int(r_edge)
            if h_edge != t_edge:
                adj_list[h_edge].append((t_edge, r_edge))

    print(f'  {scenario}/{split}: {n_pairs:,} pairs, graph {adj.nnz//2:,} edges')

    # Allocate ragged arrays
    all_nodes = []
    offsets = [0]
    n_nodes_arr = np.empty(n_pairs, dtype=np.uint16)
    heads = np.empty(n_pairs, dtype=np.uint16)
    tails = np.empty(n_pairs, dtype=np.uint16)
    rels = np.empty(n_pairs, dtype=np.uint8)
    reachable = 0

    all_edge_heads = []
    all_edge_tails = []
    all_edge_rels = []
    edge_offsets = [0]

    t0 = time.time()
    for i, (h, t, r) in enumerate(tqdm(pairs, desc=f'    {scenario}/{split}', ncols=80)):
        h, t, r = int(h), int(t), int(r)
        nodes = extract_tight_subgraph(adj, h, t, L)
        n_sub = len(nodes)

        # Đảm bảo h, t luôn có mặt (kể cả nếu unreachable — để seed embedding)
        if n_sub == 0:
            nodes = np.array([h, t], dtype=np.int64)
            n_sub = 2
        else:
            reachable += 1

        all_nodes.append(nodes.astype(np.uint16))
        offsets.append(offsets[-1] + n_sub)
        n_nodes_arr[i] = min(n_sub, 65535)
        heads[i] = h
        tails[i] = t
        rels[i] = r

        # Trích xuất các cạnh thực tế kết nối các node trong subgraph (induced subgraph)
        sub_edges_h = []
        sub_edges_t = []
        sub_edges_r = []
        node_set_lookup = set(nodes)
        for u in nodes:
            for v, r_edge in adj_list[u]:
                if v in node_set_lookup:
                    # Loại bỏ cạnh đích giữa h và t để tránh Target Leakage
                    if (u == h and v == t) or (u == t and v == h):
                        continue
                    sub_edges_h.append(u)
                    sub_edges_t.append(v)
                    sub_edges_r.append(r_edge)

        n_edges = len(sub_edges_h)
        all_edge_heads.extend(sub_edges_h)
        all_edge_tails.extend(sub_edges_t)
        all_edge_rels.extend(sub_edges_r)
        edge_offsets.append(edge_offsets[-1] + n_edges)

    elapsed = time.time() - t0

    # Concatenate ragged arrays
    all_nodes_cat = np.concatenate(all_nodes)
    offsets_arr = np.array(offsets, dtype=np.int32)
    edge_offsets_arr = np.array(edge_offsets, dtype=np.int32)
    edge_heads_arr = np.array(all_edge_heads, dtype=np.uint16)
    edge_tails_arr = np.array(all_edge_tails, dtype=np.uint16)
    edge_rels_arr = np.array(all_edge_rels, dtype=np.uint8)

    # Stats
    usable_sizes = n_nodes_arr[n_nodes_arr > 2]
    med = int(np.median(usable_sizes)) if len(usable_sizes) > 0 else 0
    p90 = int(np.percentile(usable_sizes, 90)) if len(usable_sizes) > 0 else 0
    reach_pct = reachable / n_pairs * 100

    print(f'    Done {elapsed:.1f}s | reachable={reachable}/{n_pairs} ({reach_pct:.1f}%)')
    print(f'    |V_tight|: median={med}, p90={p90}, total_nodes_stored={len(all_nodes_cat):,}')

    # Memory estimation
    node_bytes = all_nodes_cat.nbytes
    overhead_bytes = offsets_arr.nbytes + n_nodes_arr.nbytes + heads.nbytes + tails.nbytes + rels.nbytes + edge_offsets_arr.nbytes + edge_heads_arr.nbytes + edge_tails_arr.nbytes + edge_rels_arr.nbytes
    total_mb = (node_bytes + overhead_bytes) / 1024 / 1024
    print(f'    Memory: nodes={node_bytes/1024/1024:.1f}MB + overhead={overhead_bytes/1024:.1f}KB = {total_mb:.1f}MB')

    # Save
    out_path = os.path.join(out_dir, f'{scenario}_{split}_L{L}.npz')
    meta = {
        'L': L,
        'scenario': scenario,
        'split': split,
        'n_entities': n_entities,
        'n_pairs': n_pairs,
        'n_reachable': reachable,
        'median_subgraph_size': med,
        'p90_subgraph_size': p90,
        'dtype': dtype,
    }

    if dtype == 'float16':
        # Encode node IDs as float16 — lossless for integers ≤ 2048,
        # nhưng DrugBank có n_entities=34124 → float16 mất precision ở vùng cao.
        # Dùng uint16 an toàn hơn (max 65535 >> 34124).
        print(f'    Note: float16 lossy cho node ID > 2048. Dùng uint16 thay thế.')

    np.savez_compressed(
        out_path,
        offsets=offsets_arr,
        nodes=all_nodes_cat,  # uint16
        heads=heads,          # uint16
        tails=tails,          # uint16
        rels=rels,            # uint8
        edge_offsets=edge_offsets_arr,
        edge_heads=edge_heads_arr,
        edge_tails=edge_tails_arr,
        edge_rels=edge_rels_arr,
        meta=json.dumps(meta),
    )
    file_size = os.path.getsize(out_path)
    print(f'    Saved: {out_path} ({file_size/1024/1024:.2f} MB compressed)')
    return meta


# ────────────────────── CLI ──────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract tight enclosing subgraphs for EmerGNN.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # All scenarios + splits, L=3
    python extract_subgraphs.py

    # Only S1 test
    python extract_subgraphs.py --scenarios S1 --splits test

    # L=2
    python extract_subgraphs.py --L 2

    # Float16 storage (note: lossless only for node ID ≤ 2048)
    python extract_subgraphs.py --dtype float16
        """,
    )
    parser.add_argument('--scenarios', nargs='+', default=ALL_SCENARIOS,
                        choices=ALL_SCENARIOS, help='Scenarios to process')
    parser.add_argument('--splits', nargs='+', default=ALL_SPLITS,
                        choices=ALL_SPLITS, help='Splits to process')
    parser.add_argument('--L', type=int, default=3,
                        help='Max walk length (= EmerGNN args.length)')
    parser.add_argument('--dtype', default='uint16', choices=['uint16', 'int32', 'float16'],
                        help='Storage dtype for node IDs (uint16=2B/node, int32=4B, float16=2B but lossy)')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: data/subgraphs/)')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(DATA_DIR, 'subgraphs')
    os.makedirs(out_dir, exist_ok=True)

    print(f'=== Extract Tight Enclosing Subgraphs ===')
    print(f'L = {args.L}')
    print(f'Scenarios: {args.scenarios}')
    print(f'Splits:    {args.splits}')
    print(f'Dtype:     {args.dtype}')
    print(f'Output:    {out_dir}')
    print()

    # Load data
    print('Loading data...')
    t0 = time.time()
    kg, ddi, n_drugs, n_entities = load_all_data()
    print(f'  KG: {len(kg):,} triplets, {n_entities:,} entities, {n_drugs:,} drugs')
    print(f'  Loaded in {time.time()-t0:.1f}s\n')

    # Extract
    all_meta = []
    for sc in args.scenarios:
        for sp in args.splits:
            meta = extract_scenario_split(kg, ddi, sc, sp, args.L, n_entities, out_dir, args.dtype)
            if meta:
                all_meta.append(meta)
            print()

    # Summary
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'{"Scenario":<10} {"Split":<8} {"Pairs":>8} {"Reach%":>8} {"Med |V|":>10} {"p90 |V|":>10}')
    print('-' * 58)
    for m in all_meta:
        reach_pct = m['n_reachable'] / m['n_pairs'] * 100
        print(f'{m["scenario"]:<10} {m["split"]:<8} {m["n_pairs"]:>8,} {reach_pct:>7.1f}% '
              f'{m["median_subgraph_size"]:>10,} {m["p90_subgraph_size"]:>10,}')

    total_files = len(all_meta)
    total_size = sum(os.path.getsize(os.path.join(out_dir, f'{m["scenario"]}_{m["split"]}_L{m["L"]}.npz'))
                     for m in all_meta)
    print(f'\n{total_files} files, total {total_size/1024/1024:.1f} MB compressed')
    print(f'Saved to: {out_dir}/')


# ────────────────────── Loader utility ──────────────────────

class SubgraphLoader:
    """Tiện ích load subgraph đã extract, dùng trong training/eval.

    Usage:
        loader = SubgraphLoader('data/subgraphs/S1_test_L3.npz')
        nodes = loader.get_nodes(pair_idx=42)     # uint16 array
        h, t, r = loader.get_pair(pair_idx=42)
        n = loader.n_nodes[42]                     # |V_tight|

        # Batch: lấy union nodes cho nhiều pairs
        batch_nodes = loader.get_batch_nodes([0, 1, 5, 10])
    """

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.offsets = data['offsets']      # int32[N+1]
        self.nodes = data['nodes']          # uint16[]
        self.heads = data['heads']          # uint16[N]
        self.tails = data['tails']          # uint16[N]
        self.rels = data['rels']            # uint8[N]
        self.n_nodes = data['n_nodes']      # uint16[N]
        self.edge_offsets = data['edge_offsets'] # int32[N+1]
        self.edge_heads = data['edge_heads']     # uint16[]
        self.edge_tails = data['edge_tails']     # uint16[]
        self.edge_rels = data['edge_rels']       # uint8[]
        self.meta = json.loads(str(data['meta']))
        self.n_pairs = len(self.heads)
        self.n_entities = self.meta['n_entities']

    def get_nodes(self, pair_idx):
        """Trả về uint16 array of global node IDs cho pair_idx."""
        start = self.offsets[pair_idx]
        end = self.offsets[pair_idx + 1]
        return self.nodes[start:end]

    def get_edges(self, pair_idx):
        """Trả về (heads, tails, rels) của các cạnh thực tế cho pair_idx."""
        start = self.edge_offsets[pair_idx]
        end = self.edge_offsets[pair_idx + 1]
        return (
            self.edge_heads[start:end],
            self.edge_tails[start:end],
            self.edge_rels[start:end]
        )

    def get_pair(self, pair_idx):
        """Trả về (head, tail, rel)."""
        return int(self.heads[pair_idx]), int(self.tails[pair_idx]), int(self.rels[pair_idx])

    def get_batch_nodes(self, indices):
        """Union of V_tight cho nhiều pairs → sorted unique node IDs."""
        all_nodes = []
        for idx in indices:
            all_nodes.append(self.get_nodes(idx))
        if not all_nodes:
            return np.empty(0, dtype=np.uint16)
        return np.unique(np.concatenate(all_nodes))

    def get_batch_remap(self, indices):
        """Trả về (batch_nodes, heads_local, tails_local, rels) cho 1 batch.

        batch_nodes: sorted unique global IDs
        heads_local / tails_local: remapped local indices in batch_nodes
        """
        batch_nodes = self.get_batch_nodes(indices)
        global_to_local = np.full(self.n_entities, -1, dtype=np.int32)
        global_to_local[batch_nodes] = np.arange(len(batch_nodes), dtype=np.int32)

        heads_local = global_to_local[self.heads[indices]]
        tails_local = global_to_local[self.tails[indices]]
        rels_batch = self.rels[indices]
        return batch_nodes, heads_local, tails_local, rels_batch

    def __len__(self):
        return self.n_pairs

    def __repr__(self):
        return (f"SubgraphLoader({self.meta['scenario']}/{self.meta['split']}, "
                f"L={self.meta['L']}, {self.n_pairs:,} pairs, "
                f"med |V|={self.meta['median_subgraph_size']})")


if __name__ == '__main__':
    main()
