# brute_force_placer.py
"""
Brute-force placer for the 3×3 P&R task
---------------------------------------
* Reads a JSON list of training samples.
* For each sample with "placeable": true, finds one placement that satisfies
  row(u) == row(v) - 1 for every edge.
* Writes a **new** JSON file that adds a field
        "placement": [cell_idx0, cell_idx1, ...]
  where each cell_idx is in 0-8 (row*3 + col).

Usage
-----
    python brute_force_placer.py --input data.json --output data_sup.json
"""
import argparse, json, itertools

GRID_ROWS = 3
GRID_COLS = 3
CELLS = list(range(GRID_ROWS * GRID_COLS))

def valid(perm, edges):
    """perm[node] == cell index"""
    row = [c // GRID_COLS for c in perm]
    return all(row[u] == row[v] - 1 for u, v in edges)

def find_placement(N, edges):
    for perm in itertools.permutations(CELLS, N):
        if valid(perm, edges):
            return list(perm)          # first valid permutation
    return None                        # should not happen for placeable

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.load(open(args.input))
    out  = []
    for s in data:
        if s.get("placeable"):
            pl = find_placement(s["N"], s["edges"])
            if pl is None:
                raise ValueError("Marked placeable but no placement found")
            s = {**s, "placement": pl}
        out.append(s)

    json.dump(out, open(args.output, "w"), indent=2)
    print(f"Wrote {len(out)} samples → {args.output}")
