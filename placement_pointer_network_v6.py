# placement_pointer_network_v6.py  (patched)
# ------------------------------------------
# * supervised warm-up  + dense-reward actor-critic RL
# * --init  to load any checkpoint
# * edge_frac guard for empty edge list
#
# Usage example – resume RL only:
#   python placement_pointer_network_v6.py ^
#       --data   data_sup.json ^
#       --warmup 0 ^
#       --rl     60 ^
#       --batch  256 ^
#       --prefix beast_v6 ^
#       --init   beast_v6_wu03.pth
# ------------------------------------------------------

import argparse, json, sys, signal, networkx as nx
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical

# ---------------- hyper-params ----------------
MAX_NODES = 9
ROWS, COLS = 3, 3
CELLS, UNPLACED = ROWS * COLS, ROWS * COLS
EMB, HID, HEADS, LAYERS = 64, 128, 4, 4
LR, BATCH, CLIP = 2e-4, 64, 1.0
ENT_START, ENT_END = 1e-3, 1e-4
ENT_DECAY_S, ENT_DECAY_E = 8, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- data ----------------
class DS(torch.utils.data.Dataset):
    def __init__(self, path): self.d = json.load(open(path))
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]
def col(b): return b

# -------------- helpers --------------
def dist_bias(edges, N):
    G = nx.Graph(); G.add_nodes_from(range(N)); G.add_edges_from(edges)
    d = nx.all_pairs_shortest_path_length(G)
    B = torch.zeros(N, N, device=device)
    for i, nbrs in d:
        for j, l in nbrs.items():
            if l == 1: B[i, j] = 1.0
            elif l > 1: B[i, j] = -1.0
    return B

# -------------- model ---------------
class Beast(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_emb = nn.Embedding(MAX_NODES, EMB)
        self.deg = nn.Linear(1, EMB)
        self.role = nn.Embedding(2, EMB)
        self.row = nn.Embedding(1 + ROWS, EMB)      # -1 → 0
        self.proj = nn.Linear(4 * EMB, HID)
        self.enc = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=HID,
                                       nhead=HEADS,
                                       dim_feedforward=4 * HID,
                                       dropout=0.1,
                                       activation="gelu",
                                       batch_first=True)
            for _ in range(LAYERS)
        ])
        self.norm = nn.LayerNorm(HID)
        self.node_head = nn.Linear(HID, 1)
        self.cell_emb = nn.Parameter(torch.randn(CELLS + 1, HID))
        self.out_proj = nn.Linear(HID, HID)
        self.val = nn.Linear(HID, 1)

    @staticmethod
    def row_of(c): return c // COLS
    @staticmethod
    def row_idx(r): return r + 1      # -1 → 0

    def encode(self, N, edges, rows, out_deg):
        ids = torch.arange(N, device=device)
        x = torch.cat([
            self.id_emb(ids),
            self.deg(torch.tensor([[d] for d in out_deg],
                                  device=device, dtype=torch.float)),
            self.role(torch.tensor([1 if d > 0 else 0 for d in out_deg],
                                   device=device, dtype=torch.long)),
            self.row(torch.tensor([self.row_idx(r) for r in rows],
                                  device=device, dtype=torch.long))
        ], dim=-1)
        x = self.proj(x)
        bias = dist_bias(edges, N)
        for blk in self.enc: x = blk(x, src_mask=bias)
        return self.norm(x)

# -------------- supervised warm-up --------------
def supervised_step(model, sample, opt):
    if not sample.get("placeable"): return 0.0, 0
    N, edges, pl = sample["N"], sample["edges"], sample["placement"]
    out_deg = [0] * N
    preds = {v: [] for v in range(N)}
    for u, v in edges: out_deg[u] += 1; preds[v].append(u)
    rows = [-1] * N; used = set()
    loss = 0.0
    for _ in range(N):
        enc = model.encode(N, edges, rows, out_deg)
        logits = model.node_head(enc).squeeze(-1)
        elig = [rows[i] == -1 and all(rows[p] != -1 for p in preds[i])
                for i in range(N)]
        mask = torch.tensor(elig, device=device)
        logits = logits.masked_fill(~mask, -1e9)
        gt_node = max(range(N), key=lambda j: rows[j] == -1)
        loss += nn.functional.cross_entropy(logits.unsqueeze(0),
                                            torch.tensor([gt_node], device=device))
        h = enc[gt_node]
        cell_logits = (model.out_proj(h) @ model.cell_emb.t()).unsqueeze(0)
        loss += nn.functional.cross_entropy(cell_logits,
                                            torch.tensor([pl[gt_node]], device=device))
        rows[gt_node] = Beast.row_of(pl[gt_node]); used.add(pl[gt_node])
    loss.backward(); opt.step(); opt.zero_grad()
    return loss.item(), 1

# -------------- full training loop --------------
def train(model, dl, warm, rl, pre):
    opt = optim.AdamW(model.parameters(), lr=LR)
    best = -1.0
    def save(tag):
        torch.save(model.state_dict(), f"{pre}_{tag}.pth")
        print(f"[ckpt] {pre}_{tag}.pth")
    signal.signal(signal.SIGINT,
                  lambda *_: (save("INT"), sys.exit(0)))

    # ---- warm-up ----
    for ep in range(1, warm + 1):
        tot, n = 0.0, 0
        for b in dl:
            for s in b:
                l, _ = supervised_step(model, s, opt)
                tot += l; n += 1
        print(f"WU {ep}/{warm} | sup-loss {tot/n:.4f}")
        save(f"wu{ep:02d}")

    # ---- RL ----
    for ep in range(1, rl + 1):
        ent_w = ENT_START
        if ep >= ENT_DECAY_S:
            frac = min(1, (ep - ENT_DECAY_S) / ENT_DECAY_E)
            ent_w = ENT_START - frac * (ENT_START - ENT_END)
        totR, totL, steps = 0.0, 0.0, 0
        for b in dl:
            opt.zero_grad(); batch_loss = 0.0; batch_R = 0.0
            for s in b:
                N, edges, plc = s["N"], s["edges"], s["placeable"]
                preds = {v: [] for v in range(N)}
                out_deg = [0] * N
                for u, v in edges: preds[v].append(u); out_deg[u] += 1
                rows = [-1] * N; used = set()
                logps, entrs, vals = [], [], []
                success, first = True, True
                while True:
                    elig = torch.tensor(
                        [rows[i] == -1 and all(rows[p] != -1 for p in preds[i])
                         for i in range(N)], device=device)
                    if not elig.any(): success = False; break
                    enc = model.encode(N, edges, rows, out_deg)
                    v_s = model.val(enc.mean(0)); vals.append(v_s.squeeze())
                    node_logits = model.node_head(enc).squeeze(-1).masked_fill(~elig, -1e9)
                    nidx = torch.argmax(node_logits).item()
                    h = enc[nidx]
                    cell_logits = (model.out_proj(h) @ model.cell_emb.t()).clone()
                    if used: cell_logits[list(used)] = -1e9
                    for p in preds[nidx]:
                        r = rows[p]
                        for c in range(CELLS):
                            if Beast.row_of(c) != r + 1: cell_logits[c] = -1e9
                    cidx = torch.argmax(cell_logits).item()
                    logps.append(torch.log_softmax(node_logits, 0)[nidx] +
                                 torch.log_softmax(cell_logits, 0)[cidx])
                    entrs.append(-(torch.softmax(node_logits, 0) *
                                   torch.log_softmax(node_logits, 0)).sum() -
                                  (torch.softmax(cell_logits, 0) *
                                   torch.log_softmax(cell_logits, 0)).sum())
                    if cidx == UNPLACED:
                        success = (not plc) and first
                        break
                    first = False
                    rows[nidx] = Beast.row_of(cidx); used.add(cidx)
                    if all(r != -1 for r in rows): break

                # ----- dense + final reward -----
                if len(edges) == 0:
                    edge_frac = 1.0  # guard: no edges
                else:
                    edge_frac = sum(rows[u] == rows[v] - 1 for u, v in edges) / len(edges)

                r = edge_frac       # dense component
                if plc and success and edge_frac == 1.0 and all(rw != -1 for rw in rows):
                    r = 1.0         # full placement success
                if (not plc) and success:
                    r = 1.0         # correct refusal

                if logps:  # skip empty episodes
                    logps = torch.stack(logps)
                    entrs = torch.stack(entrs)
                    vals  = torch.stack(vals)
                    adv = r - vals.mean()
                    loss = -(adv.detach() * logps).sum() + 0.5 * adv.pow(2).sum() + ent_w * entrs.sum()
                    loss.backward()
                    batch_loss += loss.item()
                batch_R += r; steps += 1
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); opt.zero_grad(); totR += batch_R; totL += batch_loss
        avgR = totR / steps
        print(f"Ep {ep:3d}/{rl} | R {avgR:.3f} | L {totL/steps:.4f} | H {ent_w:.1e}")
        if ep % 2 == 0: save(f"ep{ep:03d}")
        if avgR > best: best = avgR; save("best")

# -------------- main --------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rl", type=int, default=60)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--prefix", default="beast_v6")
    p.add_argument("--init", help="path to .pth checkpoint to load")   # NEW
    a = p.parse_args()

    dl = torch.utils.data.DataLoader(
        DS(a.data), batch_size=a.batch, shuffle=True, collate_fn=col)

    model = Beast().to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device))
        print(f"Loaded weights from {a.init}")

    print("device", device)
    train(model, dl, a.warmup, a.rl, a.prefix)
