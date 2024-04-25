import json

import numpy as np
import torch
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric_temporal.signal import (
    StaticGraphTemporalSignal,
    temporal_signal_split,
)
from tqdm import tqdm

from gnn_models import STGAT, TemporalGCN, AttentionSTGCN, DConvRNN, AdaptiveGCN
from plotting import plot_prediction, plot_prediction_single, plot_prediction_full


batch_size = 1
window_size = 3
in_channels, out_channels, num_nodes, num_epochs = 3, 3, 25, 50
lr, decay, dropout = 1e-4, 5e-5, 0.1
index = {
    i: v
    for i, v in enumerate(sorted(json.load(open("sri_lanka_adj_list.json")).keys()))
}
data_file = "../Data/Datasets/sri_lanka_2013-2022_shifted.npy"


def MAPE(y_true, y_pred):
    return torch.mean(torch.abs((y_pred - y_true)) / (y_true + 1e-15) * 100)


def MAE(y_true, y_pred):
    return torch.mean(torch.abs(y_pred - y_true))


def RMSE(y_true, y_pred):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def z_norm(x, mean, std):
    return (x - mean) / std


def inverse_z_norm(x, mean, std):
    return x * std + mean


def load_adjacency_matrix(adj_file, self_loop):
    adj = json.load(open(adj_file))
    n = len(adj.keys())
    idx_map = {v: i for i, v in enumerate(sorted(adj.keys()))}
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]

    if self_loop:
        for i in range(n):
            adj_matrix[i][i] = 1

    for district in adj:
        for nei in adj[district]:
            adj_matrix[idx_map[district]][idx_map[nei]] = 1

    return adj_matrix


def create_dataset_single_shot(
    window_size=3,
    predict_ahead=3,
    self_loop=True,
    train_split=0.7,
    valid_split=0.1,
    use_disease_only=False,
):
    """
    Returns a list of temporal data of shape Nodes x Feats x Window Size
    """
    adj = load_adjacency_matrix("sri_lanka_adj_list.json", self_loop)
    edge_index = torch.tensor(
        [[x, y] for x in range(25) for y in range(25) if adj[x][y]], dtype=torch.long
    )
    x = np.nan_to_num(np.load(data_file, allow_pickle=True))
    if not use_disease_only:
        mean, std = np.mean(x), np.std(x)
    else:
        mean, std = np.mean(x[..., -6]), np.std(x[..., -6])
    x = z_norm(torch.tensor(x), mean, std)

    features = []
    targets = []
    for i in range(window_size, x.shape[0] - predict_ahead - 1):
        if use_disease_only:
            x_data = np.einsum("ij->ji", x[i - window_size : i, :, -6])
            y_data = np.einsum("ij->ji", x[i : i + predict_ahead, :, -6])
        else:
            x_data = np.einsum("ijk->jki", x[i - window_size : i, ...])
            y_data = np.einsum("ij->ji", x[i : i + predict_ahead, :, -6])
        features.append(x_data)
        targets.append(y_data)

    dataset = StaticGraphTemporalSignal(
        features=features,
        targets=targets,
        edge_index=edge_index.t().contiguous(),
        edge_weight=None,
    )
    train, test = temporal_signal_split(dataset, train_split + valid_split)
    train, valid = temporal_signal_split(
        train, (valid_split / (train_split + valid_split))
    )
    return train, valid, test, dataset, mean, std


def create_dataset_single(
    batch_size,
    window_size,
    predict_ahead,
    self_loop=True,
    device="cpu",
    train=0.7,
    val=0.1,
    use_disease_only=True,
):
    adj = load_adjacency_matrix("sri_lanka_adj_list.json", self_loop)
    edge_index = torch.tensor(
        [[x, y] for x in range(25) for y in range(25) if adj[x][y]], dtype=torch.long
    )
    x = np.nan_to_num(np.load(data_file, allow_pickle=True))
    if use_disease_only:
        mean, std = np.mean(x[:, :, -6]), np.std(x[:, :, -6])
    else:
        mean, std = np.mean(x), np.std(x)
    x = z_norm(torch.tensor(x), mean, std)
    dataset = []

    for i in range(window_size, x.shape[0] - predict_ahead):
        # F(T) x N -> N x F(T)
        if use_disease_only:
            nx = torch.swapaxes(x[i - window_size : i, :, -6], 0, 1)
            ny = torch.swapaxes(x[i : i + predict_ahead, :, -6], 0, 1)
        else:
            nx = torch.swapaxes(
                x[i - window_size : i, ...], 0, 1
            )  # T x N x F - > N X T X F
            nx = torch.swapaxes(nx, 1, 2)  # N X T X F -> N X F X T
            ny = torch.swapaxes(x[i : i + predict_ahead, :, -6], 0, 1)
        data = Data(x=nx, y=ny, edge_index=edge_index.t().contiguous())
        data.validate(raise_on_error=True)
        data.to(device)
        dataset += [data]

    drop_last = (len(dataset) % batch_size) != 0
    t_idx = int(train * len(dataset))
    v_idx = int((train + val) * len(dataset))
    train = DataLoader(dataset[:t_idx], batch_size=batch_size, drop_last=drop_last)
    val = DataLoader(dataset[t_idx:v_idx], batch_size=batch_size, drop_last=drop_last)
    test = DataLoader(dataset[v_idx:], batch_size=batch_size, drop_last=drop_last)
    full = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last)
    return train, val, test, full, mean, std


@torch.no_grad()
def infer(model, device, dataloader, mean, std, cat):
    model.eval()
    model.to(device)
    mae = rmse = mape = n = 0

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            truth = batch.y.view(pred.shape)
            if i == 0:
                y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
                y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
            truth = inverse_z_norm(truth, mean, std)
            pred = inverse_z_norm(pred, mean, std)
            y_pred[i, : pred.shape[0], :] = pred
            y_truth[i, : pred.shape[0], :] = truth
            rmse += RMSE(truth, pred)
            mae += MAE(truth, pred)
            mape += MAPE(truth, pred)
            n += 1

    rmse /= n
    mae /= n
    mape /= n

    print(f"{cat}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")
    return y_pred, y_truth


def train_model(model, train, val, device, mean, std):
    global in_channels, out_channels, num_nodes, num_epochs, lr, decay

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = torch.nn.MSELoss
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for batch in tqdm(train, desc=f"Epoch {epoch}/{num_epochs}"):
            optimizer.zero_grad()
            batch = batch.to(device)
            y_pred = model(batch)
            loss = loss_fn()(
                torch.squeeze(y_pred).float(), torch.squeeze(batch.y).float()
            )
            loss.backward()
            optimizer.step()

        print(f"Loss: {loss:.3f}")

        if not epoch % 5:
            infer(model, device, train, mean, std, "Train")
            infer(model, device, val, mean, std, "Valid")
            model.train()

    return model


@torch.no_grad()
def infer_single_shot(model, device, signals, mean, std, cat):
    model.eval()
    model.to(device)
    mae = rmse = mape = n = 0

    # TemporalTimeSignal has no __len__
    preds = []
    labels = []

    for signal in signals:
        signal = signal.to(device)
        if signal.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                if len(signal.x.shape) == 2:  # Account for single featured tensors
                    pred = model(torch.unsqueeze(signal.x, dim=1), signal.edge_index)
                    pred = torch.squeeze(pred)
                else:
                    pred = model(signal.x, signal.edge_index)
            truth = signal.y.view(pred.shape)
            truth = inverse_z_norm(truth, mean, std)
            pred = inverse_z_norm(pred, mean, std)
            preds.append(pred)
            labels.append(truth)
            rmse += RMSE(truth, pred)
            mae += MAE(truth, pred)
            mape += MAPE(truth, pred)
            n += 1

    rmse /= n
    mae /= n
    mape /= n

    print(f"{cat}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")

    preds = torch.stack(preds)
    labels = torch.stack(labels)
    return preds, labels


def train_single_shot(model, train, val, device, mean, std):
    global num_epochs, lr, decay

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = torch.nn.MSELoss
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for signal in tqdm(train, desc=f"Epoch {epoch}/{num_epochs}"):
            optimizer.zero_grad()
            signal = signal.to(device)
            if len(signal.x.shape) == 2:  # Account for single featured tensors
                y_pred = model(torch.unsqueeze(signal.x, dim=1), signal.edge_index)
            else:
                y_pred = model(signal.x, signal.edge_index)
            loss = loss_fn()(
                torch.squeeze(y_pred).float(), torch.squeeze(signal.y).float()
            )
            loss.backward()
            optimizer.step()

        print(f"Loss: {loss:.3f}")

        if not epoch % 5:
            infer_single_shot(model, device, train, mean, std, "Train")
            infer_single_shot(model, device, val, mean, std, "Valid")
            model.train()

    return model


@torch.no_grad()
def infer_ASTGCN(model, device, dataloader, mean, std, cat):

    model.eval()
    model.to(device)
    mae = rmse = mape = n = 0

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                x = torch.reshape(batch.x, (batch_size, num_nodes, -1, window_size))
                pred = model(x.float(), batch.edge_index)
                pred = torch.squeeze(pred)
            truth = batch.y.view(pred.shape)
            if i == 0:
                y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
                y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
            truth = inverse_z_norm(truth, mean, std)
            pred = inverse_z_norm(pred, mean, std)
            y_pred[i, : pred.shape[0], :] = pred
            y_truth[i, : pred.shape[0], :] = truth
            rmse += RMSE(truth, pred)
            mae += MAE(truth, pred)
            mape += MAPE(truth, pred)
            n += 1

    rmse /= n
    mae /= n
    mape /= n

    print(f"{cat}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")
    return y_pred, y_truth


def train_ASTGCN(model, train, val, device, mean, std):
    global batch_size, window_size, in_channels, out_channels, num_nodes, num_epochs, lr, decay

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = torch.nn.MSELoss
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for signal in tqdm(train, desc=f"Epoch {epoch}/{num_epochs}"):
            optimizer.zero_grad()
            signal = signal.to(device)
            x = torch.reshape(signal.x, (batch_size, num_nodes, -1, window_size))
            y_pred = model(x.float(), signal.edge_index)
            loss = loss_fn()(
                torch.squeeze(y_pred).float(), torch.squeeze(signal.y).float()
            )
            loss.backward()
            optimizer.step()

        print(f"Loss: {loss:.3f}")

        if not epoch % 5:
            infer_ASTGCN(model, device, train, mean, std, "Train")
            infer_ASTGCN(model, device, val, mean, std, "Valid")
            model.train()

    return model


@torch.no_grad()
def infer_AAGCN(model, device, dataloader, mean, std, cat):

    model.eval()
    model.to(device)
    mae = rmse = mape = n = 0

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                x = torch.reshape(batch.x, (batch_size, -1, window_size, num_nodes))
                pred = model(x.float())
                pred = torch.reshape(pred, (batch_size * num_nodes, -1))
                pred = torch.squeeze(pred)
            truth = batch.y.view(pred.shape)
            if i == 0:
                y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
                y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
            truth = inverse_z_norm(truth, mean, std)
            pred = inverse_z_norm(pred, mean, std)
            y_pred[i, : pred.shape[0], :] = pred
            y_truth[i, : pred.shape[0], :] = truth
            rmse += RMSE(truth, pred)
            mae += MAE(truth, pred)
            mape += MAPE(truth, pred)
            n += 1

    rmse /= n
    mae /= n
    mape /= n

    print(f"{cat}, MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")
    return y_pred, y_truth


def train_AAGCN(model, train, val, device, mean, std):
    global batch_size, window_size, in_channels, out_channels, num_nodes, num_epochs, lr, decay

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = torch.nn.MSELoss
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for signal in tqdm(train, desc=f"Epoch {epoch}/{num_epochs}"):
            optimizer.zero_grad()
            signal = signal.to(device)
            x = torch.reshape(signal.x, (batch_size, -1, window_size, num_nodes))
            y_pred = model(x.float())
            y_pred = torch.reshape(y_pred, (batch_size * num_nodes, -1))
            loss = loss_fn()(
                torch.squeeze(y_pred).float(), torch.squeeze(signal.y).float()
            )
            loss.backward()
            optimizer.step()

        print(f"Loss: {loss:.3f}")

        if not epoch % 5:
            infer_AAGCN(model, device, train, mean, std, "Train")
            infer_AAGCN(model, device, val, mean, std, "Valid")
            model.train()

    return model


def run_stgat():
    batch_size = 2
    train, val, test, full, m, s = create_dataset_single(
        batch_size, in_channels, out_channels, use_disease_only=True
    )
    model = STGAT(
        in_channels=in_channels,
        out_channels=out_channels,
        n_nodes=num_nodes,
        batch_size=batch_size,
        dropout=dropout,
    )
    model = train_model(model, train, val, "cpu", m, s)
    yp, yt = infer(model, "cpu", test, m, s, "Test")
    plot_prediction(yp, yt, batch_size, num_nodes, index)
    y_pred, y_truth = infer(model, "cpu", full, m, s, "Full")
    plot_prediction_full(y_pred, y_truth, batch_size, num_nodes, index)
    plot_prediction_single(y_pred, y_truth, batch_size, num_nodes, index, 9)


def run_a3tgcn():
    batch_size = 1
    use_disease_only = False
    train, val, test, full, m, s = create_dataset_single_shot(
        use_disease_only=use_disease_only
    )
    model = train_single_shot(
        TemporalGCN(1 if use_disease_only else 11, 3), train, val, "cpu", m, s
    )
    y_pred, y_truth = infer_single_shot(model, "cpu", test, m, s, "Test")
    plot_prediction(y_pred, y_truth, batch_size, num_nodes, index)
    y_pred, y_truth = infer_single_shot(model, "cpu", full, m, s, "Full")
    plot_prediction_full(y_pred, y_truth, batch_size, num_nodes, index)
    plot_prediction_single(y_pred, y_truth, batch_size, num_nodes, index, 9)


def run_astgcn():
    use_disease_only = False
    train, val, test, full, m, s = create_dataset_single(
        batch_size, in_channels, out_channels, use_disease_only=use_disease_only
    )
    model = AttentionSTGCN(
        num_nodes=25,
        num_feats=1 if use_disease_only else 11,
        window_size=3,
        predict_ahead=3,
    )
    model = train_ASTGCN(model.get_model(), train, val, "cpu", m, s)
    y_pred, y_truth = infer_ASTGCN(model, "cpu", test, m, s, "Test")
    plot_prediction(y_pred, y_truth, batch_size, num_nodes, index)
    y_pred, y_truth = infer_ASTGCN(model, "cpu", full, m, s, "Full")
    plot_prediction_full(y_pred, y_truth, batch_size, num_nodes, index)
    plot_prediction_single(y_pred, y_truth, batch_size, num_nodes, index, 9)


def run_dcrnn():
    use_disease_only = True
    train, val, test, full, m, s = create_dataset_single(
        batch_size, in_channels, out_channels, use_disease_only=True
    )
    model = DConvRNN(node_features=window_size, num_classes=3)
    model = train_model(model, train, val, "cpu", m, s)
    y_pred, y_truth = infer(model, "cpu", test, m, s, "Test")
    plot_prediction(y_pred, y_truth, batch_size, num_nodes, index)
    y_pred, y_truth = infer(model, "cpu", full, m, s, "Full")
    plot_prediction_full(y_pred, y_truth, batch_size, num_nodes, index)
    plot_prediction_single(y_pred, y_truth, batch_size, num_nodes, index, 9)


def run_aagcn():
    use_disease_only = 0
    train, val, test, full, m, s = create_dataset_single(
        batch_size, in_channels, out_channels, use_disease_only=use_disease_only
    )
    adj = load_adjacency_matrix("sri_lanka_adj_list.json", True)
    edge_index = (
        torch.tensor(
            [[x, y] for x in range(25) for y in range(25) if adj[x][y]],
            dtype=torch.long,
        )
        .t()
        .contiguous()
        .long()
    )
    model = AdaptiveGCN(1 if use_disease_only else 11, 1, num_nodes, edge_index)
    model = train_AAGCN(model.get_model(), train, val, "cpu", m, s)
    y_pred, y_truth = infer_AAGCN(model, "cpu", test, m, s, "Test")
    plot_prediction(y_pred, y_truth, batch_size, num_nodes, index)
    y_pred, y_truth = infer_AAGCN(model, "cpu", full, m, s, "Full")
    plot_prediction_full(y_pred, y_truth, batch_size, num_nodes, index)
    plot_prediction_single(y_pred, y_truth, batch_size, num_nodes, index, 9)


if __name__ == "__main__":
    run_stgat()
    run_a3tgcn()
    run_astgcn()
    run_dcrnn()
    run_aagcn()
