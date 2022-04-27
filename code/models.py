from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_add_pool, ChebConv
import torch.nn.functional as F
import torch.nn as nn
import torch


class GCN(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if not torch.is_tensor(x):
            x = x.to_dense()

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GraphSAGE_NET(torch.nn.Module):

    def __init__(self, dataset):
        super(GraphSAGE_NET, self).__init__()
        self.sage1 = SAGEConv(dataset.num_node_features, 16)  # 定义两层GraphSAGE层
        self.sage2 = SAGEConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, dataset, heads=8):
        super(GAT, self).__init__()
        self.gat1 = GATConv(dataset.num_node_features, 16, heads=8)  # 定义GAT层，使用多头注意力机制
        self.gat2 = GATConv(16*heads, dataset.num_classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)


class ChebyNet(torch.nn.Module):
    def __init__(self, dataset):
        super(ChebyNet, self).__init__()
        self.cheb1 = ChebConv(dataset.num_node_features, 16, 4)  # 定义GAT层，使用多头注意力机制
        self.cheb2 = ChebConv(16, dataset.num_classes, 4)  # 因为多头注意力是将向量拼接，所以维度乘以头数。

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.cheb1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.cheb2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, dataset):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Linear(dataset.num_node_features, 16)) # 这里用linear作为单射函数，读者可自行替换
        self.conv2 = GINConv(nn.Linear(16, dataset.num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        x = F.dropout(x, training=self.training)
        x2 = self.conv2(x, edge_index)

        x1_s = global_add_pool(x1, batch=data.batch)
        x2_s = global_add_pool(x2, batch=data.batch)
        x=torch.cat([x1_s,x2_s], -1) # 将每次迭代的节点embedding相加后进行拼接,得到图的embedding

        return F.log_softmax(x, dim=1)

