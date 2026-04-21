# Phương Án: Đưa Enclosing Subgraph Vào EmerGNN

> Kết hợp tinh thần SumGNN (extract enclosing subgraph) với cơ chế NBFNet-style propagation của EmerGNN, để giảm chi phí `generalized_rspmm` mà **không thay đổi model**.

## 1. Vấn đề hiện tại của EmerGNN

Theo [ANALYSIS.md §10.1](ANALYSIS.md) và [EDA_REPORT.md](EDA_REPORT.md):
- `generalized_rspmm` chiếm **~70–80%** thời gian training, chạy trên `(34124, 34124, 219)` cho **mọi** batch.
- Tác giả đã chọn full-graph vì **3-hop UNION** subgraph trung bình = 95% graph → extract không tiết kiệm.
- **Nhưng EDA hiện tại chưa đo INTERSECTION (enclosing)** — phép giao của 2 quả cầu L-hop. Đây là gap then chốt.

## 2. Cơ sở lý thuyết: tight enclosing subgraph

Khi seed `hiddens[h] = head_embed` rồi propagate L layer, công thức tích lũy là:
```
hiddens_L[j] = Σ_{walks h→j length=L} (∏ rel_embed_l[r_l]) ⊙ head_embed
```

### 2.1 Công thức chính xác (tight enclosing)

Ta chỉ quan tâm `hiddens_L[t]`. Một walk `h = v₀ → v₁ → … → v_L = t` độ dài đúng L. Với mỗi node trung gian `v_k`:
```
d(h, v_k) ≤ k     và     d(v_k, t) ≤ L - k
⇒  d(h, v_k) + d(v_k, t) ≤ k + (L - k) = L
```

**Tập node chính xác** cần thiết cho `hiddens_L[t]`:
```
V_tight(h, t, L) = { j : d(h, j) + d(j, t) ≤ L }
```

Ngược lại mọi node `j ∈ V_tight` đều có thể nằm trên một walk độ dài ≤ L nối `h → … → j → … → t` (vì tồn tại đường đi `h → j` dài `d(h,j)` ghép với `j → t` dài `d(j,t)`, tổng ≤ L).

### 2.2 Hệ quả then chốt

**Điều kiện cần**: `d(h, t) ≤ L`. Nếu shortest path `h ↔ t > L` thì `V_tight = ∅` — vì ∀j: `d(h,j) + d(j,t) ≥ d(h,t) > L`. Cặp drug đó **không có walk nào** độ dài ≤ L → `hiddens_L[t] = 0` → model hoàn toàn dựa vào embedding gốc (Morgan fp), không có propagated information.

**Tight ⊂ Loose**: Trong phiên bản cũ của §2 tôi dùng `V_enc = N_L(h) ∩ N_L(t)` (node nằm trong cả hai quả cầu L-hop). Tuy nhiên `V_enc ⊇ V_tight` — nhiều node thoả `d(h,j) ≤ L` và `d(t,j) ≤ L` nhưng `d(h,j) + d(t,j) > L`, tức chúng *không nằm trên bất kỳ walk nào* độ dài ≤ L. Ví dụ: node `j` cách `h` 2 hop và cách `t` 2 hop → nằm trong `V_enc` với L=3, nhưng `2+2=4 > 3` → không nằm trong `V_tight(h,t,3)`.

Theo EDA §11.6, `V_tight` **nhỏ hơn nhiều** so với `V_enc` — đây là hy vọng thực sự cho subgraph extraction.

### 2.3 Self-loop

Self-loop `(i, i, idd)` cho phép walk "đứng im" 1 bước. Một walk `h → h → A → t` có length 3, nhưng self-loop step ở `h` chỉ "lãng phí" 1 hop budget. Node `h` vẫn thoả `d(h,h)=0` → `0 + d(h,t) ≤ L`. Self-loop không làm thay đổi `V_tight` — các node ngoài `V_tight` vẫn không thể tham gia dù có self-loop.

## 3. Bước 0 — Đo trước, làm sau (PHẢI làm)

Trước khi viết code, **phải đo phân phối kích thước enclosing** trên DrugBank S0/S1/S2. Nếu enclosing trung bình > 50% graph → không bõ công. Theo trực giác từ EDA paths (median 31 distinct 2-hop paths/cặp, 266 ở 3-hop) → enclosing 2-hop nên rất nhỏ (vài chục node), 3-hop có thể vẫn nhỏ hơn nhiều so với union.

**Notebook task**: thêm cell vào [EDA_graph_paths.ipynb](EDA_graph_paths.ipynb):
```python
# Sample 500 cặp DDI, đo |N_L(h) ∩ N_L(t)| với L=2, 3
def enclosing_size(h, t, L, adj):
    Nh = bfs_ball(h, L, adj)
    Nt = bfs_ball(t, L, adj)
    return len(Nh & Nt)

# Plot phân phối; báo cáo median, p90, max
```

**Tiêu chí GO/NO-GO:**
- Median enclosing 3-hop ≤ 5,000 nodes (15% graph) → **GO**, expect ~5× speedup.
- Median ≤ 1,000 nodes (3% graph) → **GO mạnh**, expect 10–20× speedup.
- Median > 15,000 nodes → **NO-GO** cho L=3, fallback L=2.

## 4. Thiết kế: 2 phương án triển khai

### Phương án A — **Per-batch enclosing union** (đơn giản, ưu tiên thử trước)

Thay vì dựng 1 sub-KG cho mỗi cặp (overhead xếp batch lớn), ta dựng **1 sub-KG chung cho cả batch**:
```
V_batch = ⋃_{(h_i, t_i) ∈ batch}  N_L(h_i) ∩ N_L(t_i)
```
- Drug head/tail của mọi cặp trong batch đều nằm trong `V_batch`.
- Mọi node có thể tham gia tính `hiddens_L[t_i]` của bất kỳ cặp nào cũng nằm trong `V_batch`.
- Forward pass `enc_ht` chạy y nguyên, chỉ thay `KG` (full) bằng `KG_batch` (sparse, kích thước `(|V_batch|, |V_batch|, 219)`).

**Ưu**: Một lần build sub-KG cho cả batch, vẫn rất nhanh nhờ batched index. Code change tối thiểu.
**Nhược**: Nếu các cặp trong batch trải đều khắp KG, `|V_batch|` có thể phình lớn. Mitigation: **batching theo cluster** — sắp xếp cặp DDI theo gần nhau (vd dùng METIS/community detection trên drug graph) để các cặp trong 1 batch chia sẻ neighborhood.

### Phương án B — **Per-pair precomputed enclosing**, gom batch theo padded ragged tensor

Giống SumGNN: extract offline từng cặp, lưu LMDB. Mỗi cặp có sub-KG riêng, batched bằng `dgl.batch`-style block-diagonal.

**Ưu**: Subgraph nhỏ nhất có thể (chỉ enclosing thật của riêng cặp đó). Tối ưu nhất về compute.
**Nhược**: 
- Phải refactor `enc_ht` để xử lý batched sub-KG có kích thước khác nhau.
- Cần preprocessing offline (vài chục phút × 3 split).
- Mất ưu thế "shuffle KG mỗi epoch" của EmerGNN — vì sub-KG của một cặp cố định.

→ **Đề xuất**: Bắt đầu với **Phương án A**. Nếu thấy `|V_batch|` vẫn quá lớn (vd > 20% graph), chuyển sang B.

## 5. Chi tiết Phương án A — implementation steps

### Bước 1: Precompute L-hop ball cho mọi drug **một lần**

Vì 1,710 drug đều cố định, có thể build và cache 1 lần khi load data:
```python
# Thêm vào load_data.py, chạy 1 lần ở __init__
def precompute_drug_balls(self, L):
    """Trả về dict: drug_id -> set(node_id) trong L hops."""
    # Build undirected adjacency từ self.train_kg + DDI fact (dùng KG cố định nhất)
    A = build_undirected_adj(self.all_ent, self.train_kg)  # csr_matrix
    balls = {}
    for d in range(self.eval_ent):  # 1710 drugs
        ball = bfs_ball(d, L, A)    # set of node ids
        balls[d] = ball
    self.drug_balls_L = balls
```
Chi phí: 1,710 BFS × ~vài ms = vài giây. Lưu pickle.

### Bước 2: Hàm dựng sub-KG cho 1 batch

```python
def build_batch_subkg(self, head_batch, tail_batch, KG_full):
    # 1. Lấy hợp các enclosing
    V_batch = set()
    for h, t in zip(head_batch.tolist(), tail_batch.tolist()):
        V_batch |= (self.drug_balls_L[h] & self.drug_balls_L[t])
    # 2. Đảm bảo head/tail luôn có mặt (kể cả khi enclosing rỗng)
    V_batch.update(head_batch.tolist())
    V_batch.update(tail_batch.tolist())
    # 3. Remap id local
    V_list = sorted(V_batch)
    global_to_local = {g: i for i, g in enumerate(V_list)}
    # 4. Lấy slice KG: chỉ giữ các edge có cả 2 endpoint ∈ V_batch
    indices = KG_full._indices()  # (3, E)
    src, dst, rel = indices[0], indices[1], indices[2]
    mask = torch.tensor([(s.item() in V_batch) and (d.item() in V_batch) 
                         for s, d in zip(src, dst)])
    # ⚠ Bước này CHẬM nếu làm vòng for python. Cần vector hoá:
    V_tensor = torch.zeros(self.all_ent, dtype=torch.bool)
    V_tensor[list(V_batch)] = True
    mask = V_tensor[src] & V_tensor[dst]                     # vector hoá
    src_l = torch.tensor([global_to_local[s.item()] for s in src[mask]])
    dst_l = torch.tensor([global_to_local[d.item()] for d in dst[mask]])
    rel_l = rel[mask]
    # ⚠ Bước remap cũng phải vector hoá: dùng torch.searchsorted hoặc mảng map
    
    KG_sub = torch.sparse_coo_tensor(
        indices=torch.stack([src_l, dst_l, rel_l]),
        values=torch.ones(mask.sum()),
        size=(len(V_list), len(V_list), KG_full.size(2))
    ).cuda()
    
    # Remap head/tail → local
    head_local = torch.tensor([global_to_local[h.item()] for h in head_batch])
    tail_local = torch.tensor([global_to_local[t.item()] for t in tail_batch])
    return KG_sub, head_local, tail_local, len(V_list)
```

**Tối ưu vector hoá quan trọng** (tránh Python loop chậm hơn cả forward gốc):
- Dùng `torch.zeros(all_ent, dtype=bool)` làm bitmap thay vì `set` Python.
- Build remap bằng `mapping = -torch.ones(all_ent); mapping[V_tensor] = torch.arange(|V_batch|)`. Sau đó `src_l = mapping[src[mask]]`.

### Bước 3: Sửa `enc_ht` rất ít

```python
# models.py — thêm tham số n_ent động
def enc_ht(self, head, tail, KG, n_ent=None):
    if n_ent is None: n_ent = self.all_ent       # backward compat
    ...
    hiddens = torch.zeros(n_ent, len(head), self.n_dim, device='cuda')   # ★ thay np.zeros
    hiddens[head, torch.arange(len(head)).cuda()] = head_embed
    for l in range(self.L):
        hiddens = hiddens.view(n_ent, -1)
        ...
        hiddens = functional.generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
        hiddens = hiddens.view(n_ent * len(head), -1)
        ...
    tail_hid = hiddens.view(n_ent, len(tail), -1)[tail, torch.arange(len(tail))]
    ...
```
Đổi vỏn vẹn:
1. Thêm tham số `n_ent` (default = `self.all_ent`).
2. Thay `np.zeros + .cuda()` bằng `torch.zeros(..., device='cuda')` (cũng là tối ưu §10.2.B.3 trong ANALYSIS).

### Bước 4: Sửa `train()` / `evaluate()` trong `base_model.py`

```python
for h, t, r in batch_by_size(...):
    KG_sub, h_local, t_local, n_sub = self.dataloader.build_batch_subkg(h, t, KG)
    ht_embed = self.model.enc_ht(h_local, t_local, KG_sub, n_ent=n_sub)
    scores = self.model.enc_r(ht_embed)
    ...
```
DataLoader cần được pass vào BaseModel hoặc gọi trực tiếp.

### Bước 5: Thử nghiệm A/B

| Thí nghiệm | Mục đích | Metric |
|---|---|---|
| Baseline | EmerGNN gốc | F1, time/epoch |
| `+ subgraph (L=3)` | Đo speedup, kiểm tra F1 không đổi | F1 ≈ baseline ± 0.5%, time ↓ ≥ 3× |
| `+ subgraph (L=2)` | Đo trade-off cover (74% theo EDA) | F1 thấp hơn ít, time ↓ rất nhiều |
| `+ subgraph + cluster batching` | Giảm `|V_batch|` | time ↓ thêm |

**Test correctness trước khi đo speed**: chạy 1 batch nhỏ (4 cặp) qua cả full KG và sub-KG, so sánh `hiddens_L[t]` — phải gần như bằng nhau (sai số float). Đây là sanity check toán học cho chứng minh §2.

## 6. Rủi ro & mitigation

| Rủi ro | Khả năng | Mitigation |
|---|---|---|
| `|V_batch|` phình to khi batch trộn nhiều cặp xa nhau | Trung bình | Cluster batching: METIS partition trên drug-drug graph, mỗi mini-batch ⊂ 1 partition |
| Per-batch BFS lookup chậm (Python set ops) | Cao nếu code naive | Cache `drug_balls_L` thành tensor bitmap (1710 × 34124 bool) → bit-OR vector hoá trên GPU |
| 25% cặp DDI unreachable trong 4-hop (theo EDA) → enclosing rỗng | Thấp về số lượng | Ép luôn `head, tail ∈ V_batch`, sub-KG sẽ chỉ có 2 node + self-loop → propagation = 0, fallback về Morgan fp embedding (đã cover bởi `feat='M'` mode) |
| `shuffle_train` mỗi epoch làm `train_kg` thay đổi → balls invalid | Có thật | Precompute balls *trên cố định KG sinh học* (`self.train_kg` thuần KG, không đụng DDI fact). Chỉ có DDI fact thay đổi mỗi epoch — nhưng phần này nhỏ và không ảnh hưởng đáng kể đến enclosing |
| Self-loop `idd` không có trong sub-KG | Thấp | Thêm `(i,i,2*all_rel)` cho mọi `i ∈ V_batch` khi build (giống `load_graph`) |

## 7. Lộ trình triển khai (đề xuất)

| Ngày | Việc | Đầu ra |
|---|---|---|
| **D1** | Thêm cell EDA đo enclosing 2-hop, 3-hop trên 1000 cặp | Histogram + go/no-go |
| **D2** | Implement `precompute_drug_balls` + lưu pickle | `data/drug_balls_L3.pkl` |
| **D3** | Implement `build_batch_subkg` (vector hoá) + unit test correctness vs full KG | Tensor diff < 1e-5 |
| **D4** | Sửa `enc_ht`, `train`, `evaluate` để dùng sub-KG | Train 1 epoch chạy được |
| **D5** | A/B benchmark: baseline vs +subgraph (L=2, L=3) | Bảng số liệu F1 + s/epoch |
| **D6** | Nếu speedup tốt nhưng `|V_batch|` lớn: cluster batching | Thêm 1.5–2× tốc độ |
| **D7** | Đưa lên hyperparameter tuning, viết kết luận | Update ANALYSIS.md §10 |

## 8. Vì sao cách này khác / hơn SumGNN

- **Giữ nguyên model EmerGNN** — không thay sang RGCN, không cần DRNL, không cần học `entity_embed` cho protein. Chỉ thay đổi *KG* mà message passing chạy lên.
- **Chứng minh được**: `hiddens_L` trên enclosing = `hiddens_L` trên full KG. SumGNN không có bảo đảm này — nó học một model khác hẳn trên subgraph và hi vọng pattern tổng quát hoá.
- **Không cần preprocessing đắt**: chỉ cache `drug_balls`, không cần serialize subgraph mọi cặp.
- **Vẫn inductive**: drug mới chỉ cần BFS 1 lần khi xuất hiện → có ngay ball L-hop, không cần re-train.

## 9. Câu hỏi mở cần trả lời sau D1

1. **Phân phối enclosing size thực tế** thế nào? Median, p90, p99 trên DrugBank?
2. Có khác biệt giữa S0 vs S1 vs S2 không? S2 có cặp 2 drug mới → enclosing có thể rỗng nhiều hơn.
3. Cluster batching dựa trên gì? Cùng "tâm" drug? Cùng community?
4. Có cần thêm **edge sampling** trong enclosing để giảm thêm? (giống `max_nodes_per_hop` của SumGNN)

---

## Tóm tắt 1 phút

> Tận dụng việc **walk độ dài ≤ L từ h đến t bắt buộc nằm trong N_L(h) ∩ N_L(t)** để giới hạn `generalized_rspmm` về sub-KG batched. Không thay model, không thay loss, chỉ thay tensor KG. Cần trước hết đo enclosing size (EDA hiện tại chỉ đo union — đo nhầm). Nếu median enclosing ≤ 15% graph thì expect 3–10× speedup giữ nguyên F1. Implement Phương án A trước (per-batch union of enclosings), fallback Phương án B (per-pair LMDB như SumGNN) nếu A không đủ nhỏ.
