"""
placement_inference_beast.py  –  for v6 checkpoints
Greedy by default; add --beam later if you wish.
"""
import argparse, torch
from placement_pointer_network_v6 import Beast, COLS, UNPLACED, device

def row(c): return c // COLS

def preds_dict(edges, n):
    d = {v: [] for v in range(n)}
    for u, v in edges:
        d[v].append(u)
    return d

@torch.no_grad()
def greedy(model, n, edges):
    preds   = preds_dict(edges, n)
    placed  = [-1] * n
    used    = set()

    while True:
        eligible = [placed[i] == -1 and all(placed[p] != -1 for p in preds[i])
                    for i in range(n)]
        if not any(eligible):
            return "UNPLACEABLE"

        enc = model.encode(n, edges, placed, [0]*n)   # dummy out-deg
        node_logits = model.node_head(enc).squeeze(-1)
        node_logits = torch.where(torch.tensor(eligible, device=device),
                                  node_logits,
                                  torch.full_like(node_logits, -1e9))
        nidx = torch.argmax(node_logits).item()

        h = enc[nidx]
        cell_logits = (model.out_proj(h) @ model.cell_emb.t()).clone()

        if used:
            cell_logits[list(used)] = -1e9
        for p in preds[nidx]:
            r_pred = placed[p]
            for c in range(UNPLACED):          # UNPLACED left unmasked
                if row(c) != r_pred + 1:
                    cell_logits[c] = -1e9

        cidx = torch.argmax(cell_logits).item()
        if cidx == UNPLACED:
            return "UNPLACEABLE"

        placed[nidx] = row(cidx)
        used.add(cidx)

        if all(r != -1 for r in placed):
            return placed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="path to .pth checkpoint")
    args = ap.parse_args()

    model = Beast().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # quick example
    N = 7
    EDGES = [(0, 1), (0, 2), (0,3), (4,3), (3,5), (6, 5)]

    res = greedy(model, N, EDGES)
    if res == "UNPLACEABLE":
        print("Graph cannot be placed.")
    else:
        for node, r in enumerate(res):
            print(f"Node {node}: row {r}")

if __name__ == "__main__":
    main()
