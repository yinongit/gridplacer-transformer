# ------------------------- data_generator.py -------------------------
'''
Data Generator for Placement Pointer Network (3×3 grid)
========================================================

Generates random directed graphs (2–9 nodes) and ensures 80% are placeable,
20% unplaceable, per the constraint:
  For each edge u→v, row(u) == row(v) - 1.

Each JSON sample includes:
  - "N": number of nodes
  - "edges": list of [u, v]
  - "placeable": boolean

Usage:
  python data_generator.py --samples 10000 --min_nodes 2 --max_nodes 9 \
      --edge_prob 0.3 --output data.json
'''

import argparse
import json
import random
import itertools
import networkx as nx

GRID_ROWS = 3
GRID_COLS = 3
CELLS = list(range(GRID_ROWS * GRID_COLS))

def is_placeable(edges, N):
    cell_row = {c: c // GRID_COLS for c in CELLS}
    for perm in itertools.permutations(CELLS, N):
        valid = True
        for u, v in edges:
            if cell_row[perm[u]] != cell_row[perm[v]] - 1:
                valid = False
                break
        if valid:
            return True
    return False

def generate_random_graph(N, p):
    edges = []
    for u in range(N):
        for v in range(N):
            if u != v and random.random() < p:
                edges.append([u, v])
    return edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples',   type=int, default=1000)
    parser.add_argument('--min_nodes', type=int, default=2)
    parser.add_argument('--max_nodes', type=int, default=9)
    parser.add_argument('--edge_prob', type=float, default=0.3)
    parser.add_argument('--output',    type=str, default='data.json')
    args = parser.parse_args()

    random.seed(42)
    samples = []
    target_pos = int(0.8 * args.samples)
    target_neg = args.samples - target_pos
    count_pos = 0
    count_neg = 0

    while count_pos < target_pos or count_neg < target_neg:
        N = random.randint(args.min_nodes, args.max_nodes)
        edges = generate_random_graph(N, args.edge_prob)
        placeable = is_placeable(edges, N)
        if placeable and count_pos < target_pos:
            samples.append({'N': N, 'edges': edges, 'placeable': True})
            count_pos += 1
        elif not placeable and count_neg < target_neg:
            samples.append({'N': N, 'edges': edges, 'placeable': False})
            count_neg += 1
    random.shuffle(samples)

    with open(args.output, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f'Wrote {len(samples)} samples ({count_pos} placeable, {count_neg} unplaceable) to {args.output}')