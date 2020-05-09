"""
This module contains the neural model of TypeWriter
"""

from dltpy.learning.learn import top_n_fix, store_json
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Models ##########################################################################

class TWModel(nn.Module):
    """
    This is the complete neural model of TypeWriter
    """

    def __init__(self, input_size: int, hidden_size: int, aval_type_size: int, num_layers: int, output_size: int):
        super(TWModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.aval_type_size = aval_type_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm_id = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)
        self.lstm_tok = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                                bidirectional=True)
        self.lstm_cm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)

        self.linear = nn.Linear(hidden_size * 3 * 2 + self.aval_type_size, self.output_size)

    def forward(self, x_id, x_tok, x_cm, x_type):

        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_id.flatten_parameters()
        self.lstm_tok.flatten_parameters()
        self.lstm_cm.flatten_parameters()

        x_id, _ = self.lstm_id(x_id)
        x_tok, _ = self.lstm_tok(x_tok)
        x_cm, _ = self.lstm_cm(x_cm)

        # Decode the hidden state of the last time step
        x_id = x_id[:, -1, :]
        x_tok = x_tok[:, -1, :]
        x_cm = x_cm[:, -1, :]

        x = torch.cat((x_id, x_cm, x_tok, x_type), 1)

        x = self.linear(x)
        return x

    def reset_model_parameters(self):
        """
        This resets all the parameters of the model.
        It would be useful to train the model for several trials.
        """

        self.lstm_id.reset_parameters()
        self.lstm_cm.reset_parameters()
        self.lstm_tok.reset_parameters()
        self.linear.reset_parameters()

class TWModelA(TWModel):
    """
    This is the neural model of TypeWriter without available types
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(TWModelA, self).__init__(input_size, hidden_size, 0, num_layers, output_size)

    def forward(self, x_id, x_tok, x_cm, x_type):

        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_id.flatten_parameters()
        self.lstm_tok.flatten_parameters()
        self.lstm_cm.flatten_parameters()

        x_id, _ = self.lstm_id(x_id)
        x_tok, _ = self.lstm_tok(x_tok)
        x_cm, _ = self.lstm_cm(x_cm)

        # Decode the hidden state of the last time step
        x_id = x_id[:, -1, :]
        x_tok = x_tok[:, -1, :]
        x_cm = x_cm[:, -1, :]

        # Here, the available types are not concatanated.
        x = torch.cat((x_id, x_cm, x_tok), 1)

        x = self.linear(x)
        return x


class EnhancedTWModel(TWModel):
    """
    This is the enhanced model of TypeWriter with drop out
    """

    def __init__(self, input_size: int, hidden_size: int, aval_type_size: int, num_layers: int,
                 output_size: int, dropout_value : float):
        super(EnhancedTWModel, self).__init__(input_size, hidden_size, aval_type_size, num_layers,
                                              output_size)
        self.dropout = nn.Dropout(p=dropout_value)

    def forward(self, x_id, x_tok, x_cm, x_type):

        # Using dropout on input sequences
        x_id = self.dropout(x_id)
        x_tok = self.dropout(x_tok)
        x_cm = self.dropout(x_cm)

        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_id.flatten_parameters()
        self.lstm_tok.flatten_parameters()
        self.lstm_cm.flatten_parameters()

        x_id, _ = self.lstm_id(x_id)
        x_tok, _ = self.lstm_tok(x_tok)
        x_cm, _ = self.lstm_cm(x_cm)

        # Decode the hidden state of the last time step
        x_id = x_id[:, -1, :]
        x_tok = x_tok[:, -1, :]
        x_cm = x_cm[:, -1, :]

        x = torch.cat((x_id, x_cm, x_tok, x_type), 1)
        x = self.dropout(x)

        x = self.linear(x)
        return x


class BaseLineModel():
    """
    This is a naive baseline model that predicts the top n most commons types every time it is queried.
    """
    def __init__(self, most_fq_types_file):
        df_most_fq_types = pd.read_csv(most_fq_types_file)
        self.out_vec = df_most_fq_types['enc'].to_numpy()
        
    def predict(self, X):
        return np.tile(self.out_vec, (X.shape[0],1))
#################################################################################



def load_data_tensors_TW(filename, limit=-1):

    return torch.from_numpy(np.load(filename)[:limit]).float()

def load_label_tensors_TW(filename, limit=-1):

    return torch.from_numpy(np.argmax(np.load(filename), axis=1)[0:limit]).long()

def train_loop_TW(model: nn.Module, data_loader, learning_rate, epochs, tb_writer: SummaryWriter = None):

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #total_step = len(data_loader)
    #losses = np.empty(total_step * epochs)

    #i = 0
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for batch_i, (batch_id, batch_tok, batch_cm, batch_type, labels) in enumerate(data_loader):

            batch_id, batch_tok, batch_cm, batch_type = batch_id.to(device), batch_tok.to(device), batch_cm.to(device),\
                                                        batch_type.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_id, batch_tok, batch_cm, batch_type)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            #losses[i] = loss.item()
            #i += 1
            # print(f'Epoch [{epoch}/{epochs}], Batch: [{batch_i}/{total_step}], '
            #       f'Loss:{loss.item():.10f}')
        if tb_writer:
            tb_writer.add_scalar('Loss', total_loss, epoch)
        print("epoch", epoch, "loss:", total_loss)


def make_batch_prediction_TW(model: nn.Module, X_id, X_tok, X_cm, X_type, top_n=1):

    model.eval()
    with torch.no_grad():
        # Compute model output
        outputs = model(X_id, X_tok, X_cm, X_type)
        # Max for each label
        labels = np.argsort(outputs.data.cpu().numpy(), axis=1)
        labels = np.flip(labels, axis=1)
        labels = labels[:, :top_n]

        return outputs, labels


def evaluate_TW(model: nn.Module, data_loader: DataLoader, top_n=1):
    true_labels = []
    predicted_labels = []

    for i, (batch_id, batch_tok, batch_cm, batch_type, labels) in enumerate(data_loader):
        _, batch_labels = make_batch_prediction_TW(model, batch_id.to(device), batch_tok.to(device),
                                                   batch_cm.to(device), batch_type.to(device), top_n=top_n)
        predicted_labels.append(batch_labels)
        true_labels.append(labels)

    true_labels = np.hstack(true_labels)
    predicted_labels = np.vstack(predicted_labels)

    return true_labels, predicted_labels


def report_TW(y_true, y_pred, top_n, filename, result_path, params: dict):

    y_pred_fixed = top_n_fix(y_true, y_pred, top_n)

    report = classification_report(y_true, y_pred_fixed, output_dict=True)
    report = {'params': params, 'result': report}
    print("Accuracy: ", report['result']["macro avg"])
    store_json(report, f"{filename}.json", result_path)

    return report

