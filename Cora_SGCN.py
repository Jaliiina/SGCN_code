import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import subgraph
from collections import OrderedDict, deque
import random
import time

# ========== Load Cora dataset ==========
dataset = Planetoid(root='data', name='Cora', transform=NormalizeFeatures())
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# ========== Define GCN model ==========
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ========== Build adjacency list (done once globally) ==========
def build_adj_list(edge_index):
    adj = [[] for _ in range(edge_index.max().item() + 1)]
    for src, dst in edge_index.t().tolist():
        adj[src].append(dst)
        adj[dst].append(src)
    return adj

adj_list = build_adj_list(data.edge_index)

# ========== Snowball subgraph sampling ==========
def build_snowball_subgraph(seed_nodes, adj, max_nodes=100):
    visited = set(seed_nodes)
    queue = deque(seed_nodes)
    sub_nodes = list(seed_nodes)
    while queue and len(sub_nodes) < max_nodes:
        node = queue.popleft()
        for nei in adj[node]:
            if nei not in visited:
                visited.add(nei)
                sub_nodes.append(nei)
                queue.append(nei)
    return torch.tensor(sub_nodes, dtype=torch.long)

# ========== Weighted averaging of state dicts ==========
def average_state_dicts(state_dict_list, weights):
    avg_state = OrderedDict()
    total_w = sum(weights)
    for key in state_dict_list[0].keys():
        stacked = torch.stack([sd[key] * w for sd, w in zip(state_dict_list, weights)], dim=0)
        avg_state[key] = stacked.sum(dim=0) / total_w
    return avg_state

# ========== Train a subgraph model ==========
def train_on_subgraph(sub_nodes, data, base_params=None):
    sub_nodes = sub_nodes.to(device)
    sub_edge_index, _ = subgraph(sub_nodes, data.edge_index, relabel_nodes=True)
    sub_x = data.x[sub_nodes]
    sub_y = data.y[sub_nodes]

    model = GCN(sub_x.size(1), 16, dataset.num_classes).to(device)
    if base_params:
        model.load_state_dict(base_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(10):  # train 10 epochs per subgraph
        model.train()
        out = model(sub_x, sub_edge_index)
        loss = F.cross_entropy(out, sub_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on validation set to get weight
    model.eval()
    with torch.no_grad():
        out_full = model(data.x, data.edge_index)
        pred_full = out_full.argmax(dim=1)
        acc_val = (pred_full[data.val_mask] == data.y[data.val_mask]).float().mean().item()

    return model.state_dict(), acc_val

# ========== Bootstrapped training loop ==========
train_idx = data.train_mask.nonzero(as_tuple=True)[0]
base_params = None
val_acc_list = []
all_sub_times = []

for outer in range(100):  # outer loop iterations
    sub_models = []
    sub_weights = []

    for inner in range(10):  # number of subgraphs per outer loop
        seed = random.choices(train_idx.tolist(), k=10)
        sub_nodes = build_snowball_subgraph(seed, adj_list, max_nodes=100)

        start = time.time()
        state, acc_sub = train_on_subgraph(sub_nodes, data, base_params)
        duration = time.time() - start

        all_sub_times.append(duration)
        sub_models.append(state)
        sub_weights.append(acc_sub)

        print(f"Sub-model {inner+1}/10 finished | val_acc={acc_sub:.4f} | time={duration:.4f}s")

    # Weighted average
    base_params = average_state_dicts(sub_models, sub_weights)

    # Evaluate aggregated model on validation set
    model_eval = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
    model_eval.load_state_dict(base_params)
    model_eval.eval()
    with torch.no_grad():
        out = model_eval(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc_val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        val_acc_list.append(acc_val.item())
        print(f" Aggregated Validation Accuracy: {acc_val:.4f}")

# ========== Final evaluation on test set ==========
final_model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
final_model.load_state_dict(base_params)
final_model.eval()
with torch.no_grad():
    out = final_model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc_test = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    print(f"\n Final Test Accuracy: {acc_test:.4f}")

# ========== Average training time ==========
avg_sub_time = sum(all_sub_times) / len(all_sub_times)
print(f"\n Average training time per subgraph: {avg_sub_time:.4f} s")
