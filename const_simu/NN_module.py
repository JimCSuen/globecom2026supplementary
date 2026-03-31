import torch
from torch import nn
from torch.utils.data import Dataset


class TaskAllocDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Predictor_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pred_len, num_layers=1):
        super().__init__()
        self.pred_len = pred_len
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # * input shape: batch_size, seq_len, input_size
        )
        self.full_conn = nn.Linear(hidden_size, pred_len * output_size)

    def forward(self, x):
        """
        x: batch of input data sequence, size: [batch, mem_len, input_size]
        """
        # out shape: [batch, pred_len, hidden_size]
        # h_n & c_n: final hidden state & final cell state, shape: [n_layers, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x)
        out = self.full_conn(out[:, -1, :])
        out = out.view(x.size(0), self.pred_len, self.output_size)
        return out
