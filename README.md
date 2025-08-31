# SGCN_code

This repository contains implementations and experiments of **four training methods for Graph Neural Networks (GNNs)**, evaluated on the **Cora** and **Pubmed** citation network datasets.

Implemented methods:
- **SGCN** (subsampling-based GCN)
- **GraphSAGE**
- **GraphSAINT**
- **Cluster-GCN**

---

## 📌 Background
Training GNNs on large graphs is computationally expensive due to full-graph propagation.  
Subgraph-based sampling and aggregation strategies are widely adopted to improve **efficiency** while maintaining **accuracy**.  
This project systematically compares different methods under a unified experimental setting.

---

## ⚙️ Environment Setup
- Python >= 3.8
- PyTorch >= 1.12
- PyTorch Geometric >= 2.0
- matplotlib (for visualization)

Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install matplotlib
