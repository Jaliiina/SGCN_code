import argparse
import os
import os.path as osp
import torch
import torch.nn.functional as F
import time
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.nn import GraphConv
from torch_geometric.typing import WITH_TORCH_SPARSE
from torch_geometric.utils import degree

if not WITH_TORCH_SPARSE:
    quit("This example requires 'torch-sparse'")

parser = argparse.ArgumentParser()
parser.add_argument('--use_normalization', action='store_true')
args = parser.parse_args()

# 正确设置根目录，避免嵌套路径错误
root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset_name = 'Pubmed'
dataset = Planetoid(root=root, name=dataset_name)
data = dataset[0]
row, col = data.edge_index

# 添加边权重（用于归一化）
data.edge_weight = 1. / degree(col, data.num_nodes)[col]


class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


def train(model, optimizer, loader, device):
    model.train()
    model.set_aggr('add' if args.use_normalization else 'mean')

    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(model, data, device):
    model.eval()
    model.set_aggr('mean')

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())
    return accs


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 确保目录存在（尤其 GraphSAINT 要写 norm 文件）
    os.makedirs(dataset.processed_dir, exist_ok=True)

    # 禁用多进程（Windows 下 lambda 序列化会崩）
    loader = GraphSAINTRandomWalkSampler(data, batch_size=200, walk_length=2,
                                         num_steps=20, sample_coverage=80,
                                         save_dir=dataset.processed_dir,
                                         num_workers=0)
    times = []
    for epoch in range(1, 51):
        start = time.time()
        loss = train(model, optimizer, loader, device)
        accs = test(model, data, device)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
              f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}')
        times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

        
