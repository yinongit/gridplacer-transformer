# gridplacer-transformer
# Placement Beast 🧠🐍

A Transformer + Reinforcement Learning model for placing nodes on a constrained grid under structural (graph-topology) constraints.

Built for 3×3 Place & Route (P&R) tasks — but extendable.

---

## 🧩 What it does

Given:
- A directed graph of up to 9 nodes
- A 3×3 grid
- A constraint that every edge `u → v` must satisfy `row(u) = row(v) - 1`

The model learns to:
- **Assign a unique cell** to each node (if possible), or
- Declare the graph **UNPLACEABLE** when no legal assignment exists.

---

## 📦 Project Structure

| File | Description |
|------|-------------|
| `placement_pointer_network_v6.py` | Final model + training loop (supervised warm-up + RL) |
| `placement_inference_beast.py`   | Inference script for trained models |
| `data_generator_sup.py`          | Generates **labeled** supervised data using a solver |
| `data_sup.json`                  | Example dataset |
| `README.md`                      | You're reading it. |

---

## 🚀 Getting Started

1. **Install dependencies**

```bash
pip install torch networkx
