# EmerGNN - Phân Tích Chi Tiết Luồng Hoạt Động

## Mục lục
1. [Tổng quan](#1-tổng-quan)
2. [Dữ liệu đầu vào](#2-dữ-liệu-đầu-vào)
3. [Data Loading & Preprocessing](#3-data-loading--preprocessing-load_datapy)
4. [Kiến trúc Model](#4-kiến-trúc-model-modelspy)
5. [Forward Pass chi tiết](#5-forward-pass-chi-tiết---enc_ht)
6. [Loss Function](#6-loss-function)
7. [Adversarial Training](#7-adversarial-training-tùy-chọn)
8. [Training Loop](#8-training-loop)
9. [Evaluation](#9-evaluation)
10. [Phân tích Bottleneck & Tối ưu](#10-phân-tích-bottleneck--cơ-hội-tối-ưu)

---

## 1. Tổng quan

**EmerGNN** (Emergent Graph Neural Network) là mô hình GNN dự đoán **Drug-Drug Interaction (DDI)** trên Knowledge Graph.

**Bài toán:** Cho cặp thuốc `(drug_head, drug_tail)`, dự đoán loại tương tác giữa chúng (multi-class classification, 86 classes trên DrugBank).

**Ý tưởng cốt lõi:**
- Xây dựng Knowledge Graph (KG) gồm drugs + biological entities (genes, proteins) + các quan hệ
- Với mỗi cặp (head, tail), lan truyền tín hiệu (message passing) trên KG từ head → tail và tail → head
- Kết hợp thông tin lan truyền để dự đoán loại DDI

**Các scenario đánh giá:**
- **S0 (Transductive):** Cả 2 drugs đều đã thấy trong training → đánh giá khả năng dự đoán DDI mới giữa drugs đã biết
- **S1 (Inductive - 1 drug mới):** 1 drug đã biết + 1 drug chưa từng thấy → đánh giá generalization
- **S2 (Inductive - 2 drugs mới):** Cả 2 drugs đều chưa từng thấy → bài toán khó nhất

---

## 2. Dữ liệu đầu vào

### 2.1 Cấu trúc thư mục
```
data/
├── KG.txt                    # Knowledge Graph triplets (1,690,693 dòng)
├── node2id.json              # Drug name → ID (1,710 drugs, ID: 0-1709)
├── entity_drug.json          # Biological entity → ID (32,414 entities, ID: 1710-34123)
├── relation2id.json          # Relation descriptions (109 relation types)
├── DB_molecular_feats.pkl    # Morgan fingerprint features cho 1710 drugs
├── S0/                       # Transductive split
│   ├── train_ddi.txt
│   ├── valid_ddi.txt
│   └── test_ddi.txt
├── S1/                       # Inductive split (1 new drug)
│   ├── train_ddi.txt         # ~99,283 triplets
│   ├── valid_ddi.txt         # ~28,823 triplets
│   └── test_ddi.txt          # ~31,984 triplets
└── S2/                       # Inductive split (2 new drugs)
    ├── train_ddi.txt
    ├── valid_ddi.txt
    └── test_ddi.txt
```

### 2.2 Format dữ liệu

**DDI triplets** (`train_ddi.txt`, `valid_ddi.txt`, `test_ddi.txt`):
```
344     267     46        # drug_head=344, drug_tail=267, relation_type=46
144     381     46
56      694     48
```
- Mỗi dòng: `head_drug_id  tail_drug_id  relation_id`
- Drug IDs: 0-1709
- Relation IDs: 0-85 (86 loại DDI)

**KG triplets** (`KG.txt`):
```
1710    1711    86        # entity_head=1710, entity_tail=1711, relation=86
1712    1713    86
```
- Mỗi dòng: `head_entity_id  tail_entity_id  relation_id`
- Entity IDs: 1710-34123 (bao gồm genes/proteins)
- Relation IDs: 86-108 (23 loại quan hệ sinh học, nối tiếp sau 86 DDI types)

**Mapping files:**
- `node2id.json`: `{"DB04571": 0, "DB00460": 1, ...}` — 1,710 DrugBank drugs
- `entity_drug.json`: `{"Gene::801": 1710, "Gene::7428": 1711, ...}` — 32,414 biological entities
- `relation2id.json`: `{"0": "#Drug1 may increase the photosensitizing...", ...}` — 109 relation descriptions

**Morgan Fingerprint** (`DB_molecular_feats.pkl`):
- Dictionary chứa `Morgan_Features`: list 1710 vectors, mỗi vector 1024 chiều
- Biểu diễn cấu trúc phân tử của mỗi drug dưới dạng binary fingerprint

### 2.3 Thống kê tổng hợp
| Thông số | Giá trị |
|----------|---------|
| Số drugs (eval_ent) | 1,710 |
| Số biological entities | 32,414 |
| **Tổng entities (all_ent)** | **~34,124** |
| Số DDI types (eval_rel) | 86 |
| Số KG relation types | 23 |
| **Tổng relations (all_rel)** | **109** |
| Số KG triplets | 1,690,693 |

---

## 3. Data Loading & Preprocessing (`load_data.py`)

### 3.1 `__init__()` — Luồng khởi tạo

```
DataLoader(params)
    │
    ├── process_files_ddi()    ─── Đọc DDI triplets → self.triplets['train/valid/test']
    │                                Output: numpy arrays shape (N, 3) mỗi row [h, t, r]
    │
    ├── process_files_kg()     ─── Đọc KG triplets → self.train_kg, valid_kg, test_kg
    │                                Output: numpy arrays shape (M, 3)
    │                                Cập nhật entity2id, relation2id
    │
    ├── load_ent_id()          ─── Đọc mapping files → id2entity, id2relation
    │
    ├── shuffle_train()        ─── Chia DDI train thành fact + train_data, build KG
    │                                Output: self.KG (sparse tensor), self.train_data
    │
    ├── vKG = load_graph(train_DDI + valid_KG)    ─── Validation KG
    │
    └── tKG = load_graph(train_DDI + valid_DDI + test_KG)  ─── Test KG
```

### 3.2 `process_files_ddi()` — Đọc DDI

```python
Input:  3 file paths (train/valid/test)
Output: self.triplets = {
            'train': np.array shape (N_train, 3),  # [[h, t, r], ...]
            'valid': np.array shape (N_valid, 3),
            'test':  np.array shape (N_test, 3)
        }
        self.entity2id: dict {drug_id: drug_id}  (identity mapping)
        self.relation2id: dict {rel_id: rel_id}
        self.train_ent: set of drug IDs in training set
        self.eval_ent = 1710   # hardcoded số drugs
        self.eval_rel = 86     # hardcoded số DDI types
```

### 3.3 `process_files_kg()` — Đọc Knowledge Graph

```python
Input:  KG.txt path (dùng chung cho train/valid/test)
Output: self.train_kg: np.array shape (1690693, 3)
        self.valid_kg: np.array shape (1690693, 3)   # cùng data vì cùng file
        self.test_kg:  np.array shape (1690693, 3)
        self.all_ent = max entity ID + 1 ≈ 34124
        self.all_rel = max relation ID + 1 = 109
        self.ddi_in_kg: set of drug IDs có mặt trong KG
```

### 3.4 `shuffle_train()` — Chia fact/train và build KG graph

Đây là bước quan trọng, **chạy lại mỗi epoch**:

**Scenario S0 (Transductive):**
```
all_DDI_train: (N_train, 3)
    │
    ├── Random permutation
    ├── 80% đầu → fact_triplet (đưa vào KG làm thông tin đã biết)
    └── 20% còn lại → train_data (dùng để train, model phải dự đoán)

KG = load_graph(fact_triplet + train_KG)
```

**Scenario S1 (1 drug mới):**
```
all_DDI_train: (N_train, 3)
    │
    ├── Random chọn 80% drugs từ ddi_in_kg → train_ent (tập drugs "đã biết")
    │
    ├── Nếu cả h,t ∈ train_ent → fact_triplet (cả 2 drugs đã biết → cho vào KG)
    └── Nếu h ∈ train_ent XOR t ∈ train_ent → train_data (1 drug mới → cần dự đoán)
    
KG = load_graph(fact_triplet + train_KG)
```

**Scenario S2 (2 drugs mới):**
```
all_DDI_train: (N_train, 3)
    │
    ├── Random chọn 80% drugs → train_ent
    │
    ├── Nếu cả h,t ∈ train_ent → fact_triplet
    └── Nếu cả h,t ∉ train_ent → train_data (2 drugs mới → cần dự đoán)
    
KG = load_graph(fact_triplet + train_KG)
```

### 3.5 `load_graph()` — Xây dựng sparse adjacency tensor

```python
Input:  triplets: np.array shape (K, 3), mỗi row [h, t, r]
Output: sparse tensor shape (all_ent, all_ent, 2*all_rel+1)

Các bước:
1. double_triple(triplets):
   Mỗi (h, t, r) tạo ra:
   - (t, h, r)         # original direction
   - (h, t, r+n_rel)   # inverse relation
   → Output: (2K, 3)
   
2. Thêm self-loops:
   Mỗi entity i: (i, i, 2*all_rel)
   → Thêm all_ent edges, relation_id = 2*all_rel (là self-loop relation)
   
3. Tạo sparse COO tensor:
   edges: shape (2K + all_ent, 3)
   values: all ones
   → torch.sparse_coo_tensor(
       indices = edges.T,           # shape (3, num_edges)
       values  = ones(num_edges),
       size    = (all_ent, all_ent, 2*all_rel+1)
     )
```

**Ý nghĩa tensor:** `KG[i, j, r] = 1` nếu có cạnh relation r từ entity i đến entity j.

Chiều thứ 3 có kích thước `2*all_rel+1 = 2*109+1 = 219`:
- Indices 0-108: original relations
- Indices 109-217: inverse relations
- Index 218: self-loop relation

---

## 4. Kiến trúc Model (`models.py`)

### 4.1 Parameters tổng quan

```python
class EmerGNN(nn.Module):
    # Khởi tạo với:
    eval_ent = 1710          # số drugs
    eval_rel = 86            # số DDI types
    all_ent  ≈ 34124         # tổng entities
    all_rel  = 109           # tổng relation types
    L        = args.length   # số GNN layers (2 hoặc 3)
    n_dim    = args.n_dim    # embedding dimension (32 hoặc 64)
```

### 4.2 Hai chế độ Entity Feature

#### Mode `feat='E'` (Embedding) — dùng cho S0
```
ent_kg: nn.Embedding(1710, n_dim)
    - Learnable embedding cho mỗi drug
    - Lookup: ent_kg(drug_id) → vector (n_dim,)
    
Wr: nn.Linear(4*n_dim, 86)
    - Classifier: nhận concat 4 vectors → dự đoán 86 classes
```

#### Mode `feat='M'` (Morgan Fingerprint) — dùng cho S1, S2
```
ent_kg: nn.Parameter(mfeat, requires_grad=False)
    - mfeat shape: (1710, 1024) — Morgan fingerprints (frozen, không train)
    - Lookup: ent_kg[drug_id] → vector (1024,)
    
Went: nn.Linear(1024, n_dim)
    - Project fingerprint xuống n_dim dimensions
    - Went(ent_kg[drug_id]) → vector (n_dim,)
    
Wr: nn.Linear(2*n_dim, 86)
    - Classifier: nhận concat 2 vectors → dự đoán 86 classes
```

**Lý do dùng 2 mode:**
- S0 (transductive): Cả 2 drugs đã thấy → có thể dùng learnable embedding
- S1, S2 (inductive): Drug mới chưa có embedding → phải dùng molecular features (Morgan fingerprint) để biểu diễn drug mới

### 4.2.B Biểu diễn các node KHÔNG phải drug (genes, proteins, …)

Đây là phần dễ hiểu nhầm nhất của EmerGNN. Nếu chỉ đọc phần 4.2, có thể nghĩ rằng mọi entity trong KG đều có một vector đặc trưng nào đó — **không phải vậy**.

#### (1) Chỉ 1,710 drug có embedding tĩnh, 32,414 entity sinh học thì không

Nhìn lại khai báo tại `models.py`:

```python
# feat='E'
self.ent_kg = nn.Embedding(eval_ent, n_dim)          # eval_ent = 1710
# feat='M'
self.ent_kg = nn.Parameter(mfeat, requires_grad=False)  # mfeat shape (1710, 1024)
```

Cả hai chế độ đều chỉ cấp "bảng tra" có **1,710 dòng** — đúng bằng số drug. Các entity sinh học (IDs 1710–34123: genes, proteins, pathway, …) **hoàn toàn không** có embedding riêng được học/lưu ở đâu.

Vậy chúng được biểu diễn thế nào trong forward pass?

#### (2) Trong `enc_ht`, toàn bộ node khởi tạo = 0, chỉ drug truy vấn được "seed"

Nhìn lại 2 dòng đầu của mỗi chiều message passing:

```python
hiddens = torch.zeros(all_ent, batch, n_dim).cuda()   # (34124, B, n_dim)  ← TẤT CẢ = 0
hiddens[head, arange(B)] = head_embed                 # CHỈ node head được seed
```

Ý nghĩa: với mỗi truy vấn `(h, t)` trong batch, ta tạo ra một "bản đồ ẩn" `hiddens` phủ toàn bộ KG — trong đó **mọi node đều bằng vector 0**, ngoại trừ đúng một điểm duy nhất: drug `h`, được gán bằng embedding của nó. Chiều ngược `t → h` đối xứng: chỉ `t` được seed, các node còn lại = 0.

Điều này có nghĩa là: ngay trước khi message passing bắt đầu, **một protein hay một gene không có biểu diễn gì cả**. Nó chỉ là "một chỗ trống" trong đồ thị, chờ được điền vào qua các bước lan truyền.

#### (3) Representation của node trung gian "emerge" từ đường đi

Công thức lan truyền (lược bỏ batch dim cho dễ nhìn):

```
hiddens_{l+1}[j] = σ( W_l · Σ_{(i, r) ∈ N(j)}  α_{l,r}(h,t) · rel_embed_l[r]  ⊙  hiddens_l[i] )
```

Trong đó:
- `N(j)` = các cạnh `(i → j, relation r)` đi vào node `j` trong KG (đã được double + self-loop).
- `α_{l,r}(h,t)` = attention weight cho relation `r` ở layer `l`, **phụ thuộc query `(h,t)`** (chính là `relation_weight` tính từ `ht_embed`).
- `rel_embed_l[r]` = vector của relation `r` ở layer `l` — **đây là tham số học được duy nhất mang "ngữ nghĩa" của loại liên kết**.
- `⊙` = element-wise multiply (vì `mul='mul'` trong `generalized_rspmm`).

Từ đó, ta có thể suy ra bằng quy nạp:

- **Layer 0:** Chỉ những node `j` là láng giềng 1-hop của `h` mới nhận được message ≠ 0 (vì các node khác toàn bộ `hiddens_0[i] = 0`). Giá trị của chúng = tổng các `α_{0,r} · rel_embed_0[r] ⊙ head_embed` theo tất cả các cạnh `h → j`.
- **Layer 1:** Node `j` ở khoảng cách 2-hop từ `h` bắt đầu nhận message khác 0 (qua láng giềng 1-hop của nó, mà láng giềng đó vừa mới có giá trị ở layer 0).
- **Layer L:** Các node trong vùng "L-hop từ `h`" đã được điền giá trị. Giá trị `hiddens_L[j]` về bản chất là **tổng có trọng số của tất cả các đường đi (path) dài ≤ L từ `h` tới `j`**, mỗi path được nhân vào bằng tích các relation embedding trên đường đi đó (modulated bởi attention `α`).

Nói một cách hình ảnh: với một gene `j` nằm giữa `h` và `t`, biểu diễn `hiddens_L[j]` không phải "gene này là gì" mà là **"có những luồng tín hiệu nào xuất phát từ `h` mà đi qua `j`, và các luồng đó được đánh nhãn quan hệ như thế nào"**. Gene là một "giao lộ" — giá trị của nó phụ thuộc hoàn toàn vào các cạnh đi qua, không phụ thuộc ID của chính nó.

**Hệ quả quan trọng:** toàn bộ "ngữ nghĩa" mà model học được về một node trung gian nằm ở **`rel_kg[l]`** (tham số embedding của các loại relation) — chứ không nằm ở một bảng entity embedding. Vì thế trong EmerGNN chỉ có:
- 1,710 × n_dim tham số cho drug (hoặc 0 nếu feat='M').
- `L × 219 × n_dim` tham số cho relation (`rel_kg[0..L-1]`).

Không có `34,124 × n_dim` tham số entity như GCN/RGCN truyền thống.

#### (4) Điểm đọc ra (readout): `tail_hid = hiddens[tail]`

Sau `L` layer, chỉ có đúng một node được model quan tâm: drug `t`.

```python
tail_hid = hiddens.view(all_ent, B, n_dim)[tail, arange(B)]   # (B, n_dim)
```

Giá trị `tail_hid` chính là tổng hợp của **mọi đường đi ≤ L-hop từ `h` đến `t`** trong KG, với trọng số được điều khiển bởi query-dependent relation attention. Chiều ngược lại (`t → h`) đối xứng: `head_hid` là tổng hợp các đường đi từ `t` về `h`.

Cuối cùng classifier `Wr` nhìn vào `[head_hid, tail_hid]` (hoặc thêm embedding gốc với `feat='E'`) để phân loại 86 loại DDI.

#### (5) Tại sao thiết kế như vậy?

Đây là kiểu "**query-dependent subgraph reasoning**" (tương tự NBFNet, RED-GNN), khác biệt cơ bản so với GCN/GAT kinh điển:

| Khía cạnh | GCN/RGCN cổ điển | EmerGNN (NBFNet-style) |
|---|---|---|
| Node embedding ban đầu | Mỗi node có 1 vector riêng | Chỉ node truy vấn được seed, còn lại = 0 |
| Representation phụ thuộc query? | Không — embedding cố định cho mọi bài toán | Có — mỗi cặp `(h,t)` có một "bản đồ ẩn" riêng |
| Tham số chính | Entity embedding + GNN weights | **Relation embedding** + GNN weights |
| Khả năng inductive (drug mới) | Kém (không có embedding) | Tốt (không cần embedding — chỉ cần đường đi tới nó) |
| Ý nghĩa layer L | Thông tin láng giềng L-hop | Tổng các đường đi ≤ L-hop từ seed |

Chính vì thế, với một drug **hoàn toàn mới** (S1/S2):
- Chỉ cần drug đó nối vào KG qua vài quan hệ sinh học bất kỳ (ví dụ "targets gene X"), message passing đã có thể "chảy" qua nó.
- Morgan fingerprint chỉ đóng vai trò cung cấp giá trị seed — phần còn lại (representation của các protein, pathway trên đường đi) hoàn toàn tái sử dụng được từ các `rel_kg[l]` đã học.

#### (6) Lưu ý nhỏ về "self-loop relation"

Trong `load_graph` có thêm cạnh `(i, i, 2*all_rel)` cho mọi entity. Relation thứ `2*all_rel = 218` này có một vector trong `rel_kg[l]` giống mọi relation khác. Vai trò của nó là cho phép node *giữ lại* một phần giá trị của chính nó qua các layer — đóng vai trò "skip connection" trong message passing. Không có nó, node 0 vẫn mãi là 0 nếu không có láng giềng, và mọi thông tin đã tích lũy sẽ bị ghi đè hoàn toàn ở layer tiếp theo.

---

### 4.3 Shared Layers (cho mỗi GNN layer l = 0, 1, ..., L-1)

```
rel_kg[l]:          nn.Embedding(2*all_rel+1, n_dim) = Embedding(219, n_dim)
                    → Relation embedding cho từng loại quan hệ
                    
linear[l]:          nn.Linear(n_dim, n_dim)
                    → Transform sau message passing
                    
relation_linear[l]: nn.Linear(2*n_dim, 5)
                    → Nén biểu diễn cặp (head, tail) xuống 5 chiều
                    
attn_relation[l]:   nn.Linear(5, 2*all_rel+1) = Linear(5, 219)
                    → Tạo attention weight cho mỗi loại relation
```

### 4.4 Adversarial Components (tùy chọn)
```
random_layer: RandomLayer([input_dim, 86], 500)
    → Random projection cho CDAN
    
ad_net: AdversarialNetwork(500, 500)
    → Domain discriminator: 
      Linear(500,500) → ReLU → Dropout → Linear(500,500) → ReLU → Dropout → Linear(500,1) → Sigmoid
```

---

## 5. Forward Pass chi tiết - `enc_ht()`

Đây là **trái tim** của EmerGNN. Nhận cặp drugs, trả về embedding để dự đoán DDI type.

### Input
```
head: LongTensor shape (batch_size,)    — drug IDs, ví dụ [344, 144, 56]
tail: LongTensor shape (batch_size,)    — drug IDs, ví dụ [267, 381, 694]  
KG:   SparseTensor shape (all_ent, all_ent, 2*all_rel+1) ≈ (34124, 34124, 219)
```

### Bước 1: Entity Embedding

```
Mode 'E':
    head_embed = ent_kg(head)                → shape (batch, n_dim)
    tail_embed = ent_kg(tail)                → shape (batch, n_dim)
    
Mode 'M':
    head_embed = Went(ent_kg[head])          → ent_kg[head]: (batch, 1024)
                                             → Went(...): (batch, n_dim)
    tail_embed = Went(ent_kg[tail])          → shape (batch, n_dim)
```

### Bước 2: Message Passing head → tail (L layers)

Mục đích: Lan truyền tín hiệu từ head drug qua KG, xem thông tin nào đến được tail.

```python
# Khởi tạo
hiddens = zeros(all_ent, batch_size, n_dim)          # (34124, batch, n_dim)
hiddens[head[i], i] = head_embed[i] cho mỗi i        # "seed" tín hiệu tại head

ht_embed = cat(head_embed, tail_embed)                # (batch, 2*n_dim)
```

**Mỗi layer l (l = 0, 1, ..., L-1):**

```python
# --- Bước 2a: Tính Relation Attention ---
# Ý tưởng: dựa trên cặp (head, tail), xác định relation nào quan trọng

compressed = relation_linear[l](ht_embed)              # (batch, 2*n_dim) → (batch, 5)
attn_raw   = attn_relation[l](ReLU(compressed))        # (batch, 5) → (batch, 219)
relation_weight = sigmoid(attn_raw)                     # (batch, 219) — giá trị 0-1

# --- Bước 2b: Trọng số hóa Relation Embeddings ---
rel_embed = rel_kg[l].weight                            # (219, n_dim)

# Broadcasting: mỗi sample trong batch có bộ attention weights riêng
relation_input = relation_weight.unsqueeze(2) * rel_embed  
# (batch, 219, 1) * (219, n_dim) → (batch, 219, n_dim)

# Reshape cho rspmm
relation_input = relation_input.transpose(0,1).flatten(1)  # (219, batch*n_dim)

# --- Bước 2c: Generalized RSPMM (Message Passing) ---
hiddens = hiddens.view(all_ent, batch*n_dim)            # (all_ent, batch*n_dim)

hiddens = generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
# Công thức toán:
#   hiddens'[j, :] = Σ_{i,r: KG[j,i,r]≠0} KG[j,i,r] * (relation_input[r,:] ⊙ hiddens[i,:])
#
# Ý nghĩa: 
#   Node j nhận message từ tất cả neighbors i qua relation r
#   Message = relation_weight * node_embedding (element-wise multiply)
#   Tổng hợp tất cả messages (sum)
#
# Shapes:
#   KG:              sparse (all_ent, all_ent, 219)
#   relation_input:  dense  (219, batch*n_dim)
#   hiddens (input): dense  (all_ent, batch*n_dim)
#   hiddens (output):dense  (all_ent, batch*n_dim)

# --- Bước 2d: Transform ---
hiddens = hiddens.view(all_ent * batch_size, n_dim)     # (all_ent*batch, n_dim)
hiddens = linear[l](hiddens)                             # Linear transform
hiddens = ReLU(hiddens)                                   # Activation
# → hiddens reshape lại (all_ent, batch, n_dim) cho layer tiếp theo
```

**Sau L layers:**
```python
# Lấy embedding tại vị trí tail
tail_hid = hiddens.view(all_ent, batch, n_dim)[tail[i], i]  # (batch, n_dim)
```

### Bước 3: Message Passing tail → head (L layers)

Hoàn toàn đối xứng với Bước 2, nhưng seed từ tail:
```python
hiddens = zeros(all_ent, batch, n_dim)
hiddens[tail[i], i] = tail_embed[i]     # seed tại tail
# ... L layers message passing giống hệt ...
head_hid = hiddens[head[i], i]           # lấy embedding tại head → (batch, n_dim)
```

**Lưu ý quan trọng:** Relation attention weights (`relation_weight`) được tính giống hệt ở cả 2 chiều (dùng cùng `ht_embed`). Đây là redundancy có thể tối ưu.

### Bước 4: Concat thành final embedding

```
Mode 'E':
    embeddings = cat(head_embed, tail_embed, head_hid, tail_hid)
    → shape (batch, 4*n_dim)
    Gồm: original embeddings + propagated embeddings
    
Mode 'M':
    embeddings = cat(head_hid, tail_hid)
    → shape (batch, 2*n_dim)
    Chỉ dùng propagated embeddings (original fingerprints không informative đủ sau projection)
```

### Bước 5: Classification - `enc_r()`

```python
scores = Wr(embeddings)    # Linear projection
# Mode 'E': (batch, 4*n_dim) → (batch, 86)
# Mode 'M': (batch, 2*n_dim) → (batch, 86)
```

### Sơ đồ toàn bộ Forward Pass

```
INPUT: (head_ids, tail_ids, KG_sparse)
         │            │
    ┌────▼────┐  ┌────▼────┐
    │ Embed/  │  │ Embed/  │
    │ Went    │  │ Went    │
    └────┬────┘  └────┬────┘
         │            │
    head_embed   tail_embed        ← (batch, n_dim)
         │            │
         ▼            ▼
    ┌─────────────────────┐
    │  ht_embed = concat  │        ← (batch, 2*n_dim)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  DIRECTION 1: h→t   │
    │                     │
    │  seed: hiddens[h]=h_embed
    │  for l in 0..L-1:  │
    │    ┌────────────────┤
    │    │ Relation Attn  │  ht_embed → (batch, 219) weights
    │    │ × rel_embed    │  → (219, batch*n_dim) relation_input
    │    └───────┬────────┤
    │    ┌───────▼────────┤
    │    │ RSPMM on KG    │  Core message passing
    │    └───────┬────────┤
    │    ┌───────▼────────┤
    │    │ Linear + ReLU  │  Transform
    │    └────────────────┤
    │  tail_hid = hiddens[t]       ← (batch, n_dim)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  DIRECTION 2: t→h   │
    │  (symmetric)        │
    │  head_hid = hiddens[h]       ← (batch, n_dim)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  embeddings = concat │
    │  (mode-dependent)    │       ← (batch, 2*n_dim or 4*n_dim)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Wr (Linear)        │       ← (batch, 86) — DDI type scores
    └─────────────────────┘
```

---

## 6. Loss Function

### 6.1 Classification Loss (base_model.py, dòng 108-112)

```python
p_score = scores[arange, r]                              # score của class đúng
n_score = scores                                          # all scores
max_n   = max(n_score, dim=1, keepdim=True)[0]           # numerical stability
loss    = -p_score + max_n + log(sum(exp(n_score - max_n)))
loss    = loss.sum()
```

Đây chính là **Softmax Cross-Entropy Loss** viết dưới dạng numerically stable:
```
L = -log(softmax(scores)[correct_class])
  = -p_score + log(Σ exp(scores))
  = -p_score + max_n + log(Σ exp(scores - max_n))   ← log-sum-exp trick
```

### 6.2 Adversarial Loss (tùy chọn, loss.py)

**CDAN Loss** (Conditional Domain Adversarial Network):
```python
# Kết hợp feature + prediction
softmax_output = softmax(scores).detach()     # (2*batch, 86) — detach gradient
feature = ht_embed                             # (2*batch, 2*n_dim hoặc 4*n_dim)

# Random projection
random_out = RandomLayer([feature, softmax_output])   # → (2*batch, 500)

# Domain discrimination
ad_out = AdversarialNetwork(random_out)               # → (2*batch, 1) — sigmoid output

# BCE loss với domain labels
dc_target = [1]*batch + [0]*batch    # source=1, target=0
ad_loss = BCELoss(ad_out, dc_target)
```

**Gradient Reversal Layer (GRL):**
```python
# Trong AdversarialNetwork.forward():
coeff = 2/(1+exp(-10*iter/max_iter)) - 1    # tăng dần từ 0→1
x.register_hook(lambda grad: -coeff * grad)  # đảo gradient khi backprop
```
GRL đảo ngược gradient: model học features không phân biệt được domain, giúp generalize từ source → target.

### 6.3 Total Loss
```
total_loss = classification_loss + adversarial_weight * ad_loss
```

---

## 7. Adversarial Training (tùy chọn)

Khi flag `--adversarial` được bật (chủ yếu cho S1/S2 inductive setting):

### Mục đích
Domain adaptation: Giúp model generalize từ drugs đã biết (source domain) sang drugs mới (target domain).

### Cơ chế
1. **Source data:** `train_pos` — DDI triplets từ dataset hiện tại (S1 hoặc S2)
2. **Target data:** `train1_pos` — DDI triplets từ S1 dataset (valid + test sets)
3. Hai nguồn data được balance (repeat/truncate) và train song song
4. Discriminator (`ad_net`) cố phân biệt source vs target
5. Feature extractor (`enc_ht`) học features **không** phân biệt được domain (qua GRL)

### Network Components

```
RandomLayer:
    Input:  [feature (2*n_dim), prediction (86)]
    Random matrices: (2*n_dim, 500) và (86, 500) — frozen random
    Output: element-wise product of two random projections → (batch, 500)

AdversarialNetwork:
    Linear(500, 500) → ReLU → Dropout(0.5)
    → Linear(500, 500) → ReLU → Dropout(0.5)
    → Linear(500, 1) → Sigmoid
    Output: probability ∈ [0,1] — source or target domain
```

---

## 8. Training Loop

### Entry Point: `evaluate.py` (thực chất là main training script)

```python
# 1. Setup
dataloader = DataLoader(args)          # Load tất cả data, build graphs
model = BaseModel(eval_ent, eval_rel, args)   # Khởi tạo EmerGNN

# 2. Mỗi epoch:
for e in range(n_epoch):               # default: 100 epochs
    dataloader.shuffle_train()         # ★ Tái chia fact/train, REBUILD KG
    KG = dataloader.KG                 # KG mới cho epoch này
    train_pos = dataloader.train_data  # Training data mới
    
    model.train(train_pos, None, train1_pos, None, KG)   # Train 1 epoch
    
    if (e+1) % epoch_per_test == 0:    # default: mỗi 5 epochs
        v_f1, v_acc, v_kap, _ = model.evaluate(valid_pos, None, vKG)
        t_f1, t_acc, t_kap, _ = model.evaluate(test_pos, None, tKG)
        scheduler.step(v_f1)           # Reduce LR on plateau
```

### Trong `model.train()` (1 epoch):

```python
for batch (h, t, r) in batches of size n_batch:
    # Forward
    ht_embed = model.enc_ht(h, t, KG)    # ★ BOTTLENECK
    scores   = model.enc_r(ht_embed)      # Classification
    
    # Loss
    loss = softmax_cross_entropy(scores, r)
    if adversarial:
        loss += weight * CDAN_loss(...)
    
    # Backward
    loss.backward()
    optimizer.step()
```

### Hyperparameters mặc định theo scenario

| Parameter | S0 | S1 | S2 |
|-----------|-----|-----|-----|
| lr | 0.003 | 0.003 | 0.003 |
| lamb (weight decay) | 1e-8 | 1e-8 | 1e-4 |
| n_batch | 64 | 32 | 32 |
| n_dim | 32 | 64 | 32 |
| length (L) | 3 | 3 | 3 |
| feat | 'E' | 'M' | 'M' |
| test_batch_size | 16 | 16 | 16 |

---

## 9. Evaluation

### Metrics
```python
# Trong base_model.evaluate():
pred  = argmax(softmax(scores), axis=1)    # Predicted class
label = ground_truth_relations              # True class

accuracy = sum(pred == label) / len(pred)
f1       = f1_score(label, pred, average='macro')    # Macro F1
kappa    = cohen_kappa_score(label, pred)             # Cohen's Kappa
accuracy_per_class = diagonal(confusion_matrix) / row_sums   # Per-class accuracy
```

### Evaluation Flow
```
Mỗi epoch_per_test epochs:
    valid: evaluate(valid_pos, vKG) → v_f1, v_acc, v_kappa
    test:  evaluate(test_pos, tKG)  → t_f1, t_acc, t_kappa
    
    if v_f1 > best → save model
    scheduler.step(v_f1)
```

---

## 10. Phân tích Bottleneck & Cơ hội Tối ưu

### 10.1 Phần tốn thời gian nhất (xếp theo mức độ ảnh hưởng)

#### #1: `generalized_rspmm()` — ~70-80% tổng thời gian

```
Gọi: 2 chiều (h→t, t→h) × L layers × mỗi batch = 2*L = 6 lần/batch (L=3)
Kích thước: 
    KG sparse: (34124, 34124, 219)  — ~1.69M non-zero entries × 2 (double) + 34K (self-loop)
    relation_input: (219, batch*n_dim) = (219, 32*64) = (219, 2048)
    hiddens: (34124, 2048)

Tại sao chậm:
- Sparse-dense multiply trên tensor 3D rất lớn
- Phải iterate qua ~3.4M non-zero entries cho mỗi lần gọi
- 6 lần gọi/batch × hàng trăm batches/epoch = hàng nghìn lần/epoch
- Đây là phép toán trên TOÀN BỘ KG (34K nodes), không chỉ các drugs trong batch
```

#### #2: `shuffle_train()` — ~10-15% tổng thời gian

```
Gọi: 1 lần/epoch
Chi phí:
- Chia lại fact/train_data: O(N_train) — nhẹ
- load_graph(): Tạo sparse CUDA tensor mới
  - double_triple: 2*K triples → numpy operations
  - Thêm self-loops: concatenate
  - Tạo torch.sparse_coo_tensor trên CUDA
  
Tại sao chậm:
- Mỗi epoch tạo mới sparse tensor kích thước (34124, 34124, 219)
- Transfer data CPU → GPU
- Memory allocation + deallocation
```

#### #3: Zero tensor allocation trong `enc_ht()` — ~5%

```python
hiddens = torch.FloatTensor(np.zeros((n_ent, len(head), self.n_dim))).cuda()
# Gọi 2 lần/forward pass (h→t và t→h)
# Size: (34124, batch_size, n_dim) = (34124, 32, 64) ≈ 560MB mỗi lần
# Tạo numpy array → convert to tensor → move to CUDA
```

### 10.2 Phần có thể chạy trước / tối ưu

#### A. Đã tối ưu sẵn (không cần thay đổi)

| Phần | Lý do |
|------|-------|
| `vKG` (validation graph) | Build 1 lần duy nhất ở `__init__`, dùng lại mọi epoch |
| `tKG` (test graph) | Build 1 lần duy nhất ở `__init__`, dùng lại mọi epoch |
| Morgan fingerprints | Load 1 lần từ pickle, dùng `requires_grad=False` |

#### B. Có thể pre-compute (dễ thực hiện)

**1. Pre-compute `Went(mfeat)` cho toàn bộ drugs (feat='M'):**
```python
# Hiện tại: tính lại mỗi batch
head_embed = self.Went(self.ent_kg[head])   # mỗi batch

# Tối ưu: tính 1 lần sau mỗi optimizer step
self.drug_embeds = self.Went(self.ent_kg)   # (1710, n_dim) — tính 1 lần
head_embed = self.drug_embeds[head]          # chỉ lookup
```
*Tiết kiệm:* ~1710 × batch_count matrix multiplies/epoch → 1 matrix multiply/epoch

**2. Cache relation attention weights:**
```python
# Hiện tại: tính 2 lần (h→t và t→h) với CÙNG input
# h→t:
relation_weight = sigmoid(attn_relation[l](ReLU(relation_linear[l](ht_embed))))
# t→h:  
relation_weight = sigmoid(attn_relation[l](ReLU(relation_linear[l](ht_embed))))  # GIỐNG HỆT!

# Tối ưu: tính 1 lần, dùng lại
for l in range(L):
    cached_weights[l] = sigmoid(attn_relation[l](ReLU(relation_linear[l](ht_embed))))
# Dùng cached_weights[l] cho cả 2 chiều
```
*Tiết kiệm:* L forward passes qua 2 linear layers/batch → giảm 50%

**3. Dùng `torch.zeros()` thay vì `np.zeros()` + convert:**
```python
# Hiện tại (chậm):
hiddens = torch.FloatTensor(np.zeros((n_ent, len(head), self.n_dim))).cuda()
# numpy array (CPU) → torch tensor (CPU) → .cuda() (GPU): 2 lần copy

# Tối ưu:
hiddens = torch.zeros(n_ent, len(head), self.n_dim, device='cuda')
# Allocate trực tiếp trên GPU: 0 lần copy
```
*Tiết kiệm:* Giảm 2 memory copy operations × 2 lần/forward × mỗi batch

#### C. Có thể tối ưu sâu hơn (cần thay đổi logic)

**4. Fix KG cho S0:**
```
S0 dùng random 80/20 split → mỗi epoch shuffle lại → KG thay đổi
Nếu fix random seed hoặc fix split: KG không đổi → không cần rebuild
Trade-off: Giảm data augmentation nhưng tiết kiệm toàn bộ load_graph()/epoch
```

**5. Subgraph sampling thay vì full-graph:**
```
Hiện tại: RSPMM trên TOÀN BỘ 34K nodes, dù batch chỉ có 32-64 drugs
Tối ưu:   Extract L-hop subgraph quanh (head, tail) → RSPMM trên subgraph nhỏ hơn nhiều
Trade-off: Phức tạp implementation, có thể mất accuracy nếu cut sai
```

**6. Pre-compute relation_input per layer:**
```
rel_embed = rel_kg[l].weight              # (219, n_dim) — không đổi trong 1 epoch
relation_input phụ thuộc batch → không cache được giữa batches
Nhưng có thể cache rel_embed * attention cho cả batch trước khi rspmm
```

### 10.3 Bảng tổng hợp

| Phần | Thời gian | Pre-compute? | Độ khó | Tiết kiệm ước tính |
|------|-----------|-------------|--------|-------------------|
| `generalized_rspmm` | ~70-80% | Không (phụ thuộc batch) | - | - |
| `shuffle_train` + `load_graph` | ~10-15% | Có (fix split cho S0) | Thấp | ~10-15%/epoch |
| `np.zeros → cuda` allocation | ~5% | Có (dùng `torch.zeros(device='cuda')`) | Rất thấp | ~5%/forward |
| `Went(mfeat)` mỗi batch | ~2-3% | Có (cache sau optimizer step) | Thấp | Minor |
| Relation attention (tính 2 lần) | ~2-3% | Có (cache 1 lần) | Rất thấp | ~1-2% |
| **Subgraph sampling** | Thay đổi core | Có (thay đổi kiến trúc) | **Cao** | **Có thể ~50%+** |

### 10.4 Nhận xét thiết kế

**Điểm mạnh:**
- Message passing 2 chiều (h→t, t→h) capture được asymmetric relationships
- Relation attention cho phép model học relation nào quan trọng cho từng cặp drug
- Morgan fingerprint (feat='M') cho phép inductive learning trên drug mới
- Adversarial training giúp domain adaptation cho S1/S2 setting

**Điểm yếu/hạn chế:**
- Message passing trên **toàn bộ KG** cho mỗi batch → chi phí O(|E| × batch × n_dim) mỗi layer
- `shuffle_train()` rebuild KG mỗi epoch → overhead lớn không cần thiết (nhất là S1/S2 nơi split logic cố định nếu train_ent cố định)
- Nhiều unnecessary CPU↔GPU transfers (`np.zeros` → `cuda`)
- Relation attention được tính trùng lặp 2 lần cho cùng input
