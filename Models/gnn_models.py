import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric_temporal import A3TGCN, ASTGCN, AAGCN
from torch_geometric_temporal.nn.recurrent import DCRNN


class AdaptiveGCN(torch.nn.Module):
    def __init__(self, num_features, out_channels, num_nodes, edge_index):
        super(AdaptiveGCN, self).__init__()
        self.aagcn = AAGCN(
            in_channels=num_features,
            out_channels=out_channels,
            edge_index=edge_index,
            num_nodes=num_nodes,
            stride=1,
            residual=True,
            adaptive=False,
            attention=True,
        )

    def get_model(self):
        return self.aagcn


class AttentionSTGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_feats, window_size, predict_ahead):
        super(AttentionSTGCN, self).__init__()
        self.gcn = ASTGCN(
            nb_block=2,
            K=3,
            in_channels=num_feats,
            nb_chev_filter=64,
            nb_time_filter=64,
            time_strides=1,
            len_input=window_size,
            num_for_predict=predict_ahead,
            num_of_vertices=num_nodes,
        )

    def get_model(self):
        return self.gcn


class TemporalGCN(torch.nn.Module):
    def __init__(self, num_feats, window_size):
        super(TemporalGCN, self).__init__()
        self.tgcn = A3TGCN(in_channels=num_feats, out_channels=32, periods=window_size)
        self.linear = torch.nn.Linear(32, window_size)

    def forward(self, x, edge_index):
        x = self.tgcn(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        return x


class STGAT(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, n_nodes, batch_size, heads=8, dropout=0.0
    ):
        super(STGAT, self).__init__()
        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        h_1 = 64
        h_2 = 64

        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=in_channels,
            heads=heads,
            dropout=dropout,
            concat=False,
        )
        self.lstm1 = torch.nn.LSTM(
            input_size=self.n_nodes, hidden_size=h_1, num_layers=1
        )
        self.lstm2 = torch.nn.LSTM(input_size=h_1, hidden_size=h_2, num_layers=1)

        for (n1, p1), (n2, p2) in zip(
            self.lstm1.named_parameters(), self.lstm2.named_parameters()
        ):
            if "bias" in n1:
                torch.nn.init.constant_(p1, 0.0)
            if "bias" in n2:
                torch.nn.init.constant_(p2, 0.0)
            if "weight" in n1:
                torch.nn.init.xavier_uniform_(p1)
            if "weight" in n2:
                torch.nn.init.xavier_uniform_(p2)

        self.linear = torch.nn.Linear(h_2, self.n_nodes * self.n_pred)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat(torch.tensor(x, dtype=torch.float), edge_index)
        x = F.dropout(x, self.dropout)
        x = torch.reshape(
            x,
            (data.num_graphs, int(data.num_nodes / self.batch_size), data.num_features),
        )
        x = torch.movedim(x, 2, 0)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.linear(torch.squeeze(x[-1, :, :]))
        nx = x.shape[0]
        x = torch.reshape(x, (nx, self.n_nodes, self.n_pred))
        x = torch.reshape(x, (nx * self.n_nodes, self.n_pred))
        return x


class DConvRNN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.recurrent = DCRNN(node_features, 64, K=32)
        self.linear = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if len(x.shape) == 3:
            x = torch.reshape(x, (x.shape[0], -1))
        h = self.recurrent(x.float(), edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h
