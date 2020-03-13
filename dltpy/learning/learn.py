import os
import pickle
import time
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils import data
from torch.utils.data import DataLoader
from typing import Tuple

from dltpy import config

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_model_parameters(model: nn.Module) -> int:
    """
    Count the amount of parameters of a model
    :param model:
    :return:
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def store_model(model: nn.Module, filename: str, model_dir=config.MODEL_DIR) -> None:
    """
    Store the model to a pickle file
    :param model: the model itself
    :param filename: name of the file
    :param model_dir: directory in which to write to file to
    :return:
    """
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, filename), 'wb') as f:
        pickle.dump(model, f)

def store_json(model: nn.Module, filename: str, model_dir=config.MODEL_DIR) -> None:
    """
    Store the model as a json file.
    :param model: the model itself
    :param filename: name of the file
    :param model_dir: directory in which to write to file to
    :return:
    """
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, filename), 'w') as f:
        f.write(json.dumps(model))

def load_model(filename: str, model_dir=config.MODEL_DIR) -> nn.Module:
    """
    Load the model from a pickle.
    :param filename: name of the file
    :param model_dir: directory in which the file is located
    :return:
    """
    with open(os.path.join(model_dir, filename), 'rb') as f:
        return pickle.load(f)

class BiRNN(nn.Module):
    """
    The BiRNN represents the implementation of the Bidirectional RNN model
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional=True) -> None:
        super(BiRNN, self).__init__()

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1000)

    def forward(self, x):
        x = self.dropout(x)

        # Forward propagate LSTM
        # Out: tensor of shape (batch_size, seq_length, hidden_size*2)
        x, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        x = x[:, -1, :]

        # Output layer
        x = self.linear(x)

        return x

class GRURNN(nn.Module):
    """
    The GRURNN represents the implementation of the GRU RNN model
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional=True) -> None:
        super(GRURNN, self).__init__()

        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=0.2)

        self.lstm = nn.GRU(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=bidirectional)

        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1000)

    def forward(self, x):
        x = self.dropout(x)

        # Forward propagate LSTM
        # Out: tensor of shape (batch_size, seq_length, hidden_size*2)
        x, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        x = x[:, -1, :]

        # Output layer
        x = self.linear(x)

        return x

def load_dataset(X, y, batch_size: int, split=0.8) -> Tuple:
    """
    Load and return a specific dataset
    :param X: x input part of the dataset
    :param y: y input part of the dataset
    :param batch_size: size to use for the batching
    :param split: amount of data to split (between 0 and 1)
    :return: tuple consisting out of a training and test set
    """
    train_data = torch.utils.data.TensorDataset(X, y)

    train_size = int(split * len(train_data))
    test_size = len(train_data) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def load_data_tensors(filename_X: str, filename_y: str, limit: int) -> Tuple:
    """
    Load the tensor dataset
    :param filename_X: x input part of the dataset
    :param filename_y: y input part of the dataset
    :param limit: max amount of y data to load in
    :return: Tuple (X,y) consisting out of the tensor dataset
    """
    X = torch.from_numpy(np.load(filename_X)[0:limit]).float()
    y = torch.from_numpy(np.argmax(np.load(filename_y), axis=1)[0:limit]).long()
    return X, y

def make_batch_prediction(model: nn.Module, X, top_n=1):
    model.eval()
    with torch.no_grad():
        # Compute model output
        outputs = model(X)
        # Max for each label
        labels = np.argsort(outputs.data.cpu().numpy(), axis=1)
        labels = np.flip(labels, axis=1)
        labels = labels[:, :top_n]
        return outputs, labels

def evaluate(model: nn.Module, data_loader: DataLoader, top_n=1):
    true_labels = []
    predicted_labels = []

    for i, (batch, labels) in enumerate(data_loader):
        _, batch_labels = make_batch_prediction(model, batch.to(device), top_n=top_n)
        predicted_labels.append(batch_labels)
        true_labels.append(labels)

    true_labels = np.hstack(true_labels)
    predicted_labels = np.vstack(predicted_labels)

    return true_labels, predicted_labels

def top_n_fix(y_true, y_pred, n):
    best_predicted = np.empty_like(y_true)
    for i in range(y_true.shape[0]):
        if y_true[i] in y_pred[i, :n]:
            best_predicted[i] = y_true[i]
        else:
            best_predicted[i] = y_pred[i, 0]

    return best_predicted

def train_loop(model: nn.Module, data_loader: DataLoader, model_config: dict, model_store_dir,
               save_each_x_epochs=25):
    model.train()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    losses = []

    # Train the model
    total_step = len(data_loader)
    losses = np.empty(total_step * model_config['num_epochs'])
    i = 0
    for epoch in range(1, model_config['num_epochs'] + 1):
        for batch_i, (batch, labels) in enumerate(data_loader):
            batch = batch.to(device)
            labels = labels.to(device)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            losses[i] = loss.item()
            i += 1

        print(f'Epoch [{epoch}/{model_config["num_epochs"]}], Batch: [{batch_i}/{total_step}], '
              f'Loss:{loss.item():.10f}')
        if device == 'cuda':
            print(f"Cuda v-memory allocated {torch.cuda.memory_allocated()}")

        if epoch % save_each_x_epochs == 0 or (epoch == model_config['num_epochs']):
            print("Storing model!")
            store_model(model, f"model_{model.__class__.__name__}_e_{epoch}_l_{loss.item():0.10f}.h5",
                        model_dir=os.path.join(config.MODEL_DIR, model_store_dir))

    return losses

def load_m1():
    model_config = {
        'sequence_length': 55,
        'input_size': 14,  # The number of expected features in the input `x`
        'hidden_size': 14,
        'num_layers': 2,
        'batch_size': 256,
        'num_epochs': 100,
        'learning_rate': 0.002,
        'bidirectional': True
    }
    # Load the model
    model = BiRNN(model_config['input_size'], model_config['hidden_size'],
                  model_config['num_layers'], model_config['bidirectional']).to(device)
    return model, model_config

def load_m2():
    model_config = {
        'sequence_length': 55,
        'input_size': 14,  # The number of expected features in the input `x`
        'hidden_size': 10,
        'num_layers': 1,
        'batch_size': 256,
        'num_epochs': 100,
        'learning_rate': 0.002,
        'bidirectional': False
    }
    # Load the model
    model = GRURNN(model_config['input_size'], model_config['hidden_size'],
                   model_config['num_layers'], model_config['bidirectional']).to(device)
    return model, model_config

# Model C
def load_m3():
    model_config = {
        'sequence_length': 55,
        'input_size': 14,  # The number of expected features in the input `x`
        'hidden_size': 128,
        'num_layers': 1,
        'batch_size': 256,
        'num_epochs': 25,
        'learning_rate': 0.002,
        'bidirectional': True
    }
    # Load the model
    model = BiRNN(model_config['input_size'], model_config['hidden_size'],
                  model_config['num_layers'], model_config['bidirectional']).to(device)
    return model, model_config

def load_m4():
    model_config = {
        'sequence_length': 55,
        'input_size': 14,  # The number of expected features in the input `x`
        'hidden_size': 20,
        'num_layers': 1,
        'batch_size': 256,
        'num_epochs': 100,
        'learning_rate': 0.002,
        'bidirectional': True
    }
    # Load the model
    model = BiRNN(model_config['input_size'], model_config['hidden_size'],
                  model_config['num_layers'], model_config['bidirectional']).to(device)
    return model, model_config

def get_datapoints(dataset: str) -> Tuple[str, str, str, str]:
    base = "./output/vectors/"
    return base + "return_datapoints_x.npy", base + "return_datapoints_y.npy", base + "param_datapoints_x.npy", \
           base + "param_datapoints_y.npy"


def report(y_true, y_pred, top_n, total_dps, filename: str):
    # Fix the predictions if the true value is in top-n predictions
    y_pred_fixed = top_n_fix(y_true, y_pred, top_n)

    pred_corr = np.count_nonzero(np.equal(y_true, y_pred_fixed))

    precision = pred_corr / y_pred.shape[0]
    recall = pred_corr / total_dps
    f1_score = (2 * precision * recall) / (precision + recall)

    #print("p:", precision)
    #print("r:", recall)
    #print("f1:", f1_score)

    report = {"weighted avg": {"precision": precision, "recall": recall, "f1-score": f1_score}}
    print(report)
    # Computation of metrics
    # report = classification_report(y_true, y_pred_fixed, output_dict=True)
    # print(report["weighted avg"])
    # store_model(report, f"{filename}.pkl", "./output/reports/pkl")
    store_json(report, f"{filename}.json", "./output/reports/json")

    return report


def report_loss(losses, filename: str):
    store_model(losses, f"{filename}.pkl", "./output/reports/pkl")
    store_json({"loss": list(losses)}, f"{filename}.json", "./output/reports/json")


def main():

    print(f"-- Using {device} for training.")

    top_n_pred = [1, 2, 3]
    models = [load_m3]
    # datasets = ["2_cf_cr_optional", "3_cp_cf_cr_optional", "4_complete_without_return_expressions"]
    n_repetitions = 3

    # for dataset in datasets:
    dataset = "Complete"
    # Load data
    RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y = get_datapoints(dataset)
    # print(f"-- Loading data: {dataset}")
    Xr, yr = load_data_tensors(RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, limit=-1)
    Xp, yp = load_data_tensors(PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y, limit=-1)
    X = torch.cat((Xp, Xr))
    y = torch.cat((yp, yr))

    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])

    for load_model in models:
        for i in range(n_repetitions):
            model, model_config = load_model()

            print(f"-- Model Loaded: {model} with {count_model_parameters(model)} parameters.")

            train_loader, test_loader = load_dataset(X, y, model_config['batch_size'], split=0.8)

            # Start training
            losses = train_loop(model, train_loader, model_config, model_store_dir=f"{load_model.__name__}/{dataset}/{i}"+str(int(time.time())))

            # print("-- Loading model")
            # model = load_model('1571306801/model_BiRNN_e_9_l_1.8179169893.h5')

            # Evaluate model performance
            y_true, y_pred = evaluate(model, test_loader, top_n=max(top_n_pred))

            # If the prediction is "other" - ignore the result
            idx_of_other = pickle.load(open(f'./output/ml_inputs/label_encoder.pkl', 'rb')).transform(['other'])[0]
            idx = (y_true != idx_of_other) & (y_pred[:, 0] != idx_of_other)

            for top_n in top_n_pred:
                filename = f"{load_model.__name__}_{dataset}_{i}_{top_n}"
                report(y_true, y_pred, top_n, filename)
                report(y_true[idx], y_pred[idx], top_n, f"{filename}_filtered")

            report_loss(losses, f"{load_model.__name__}_{dataset}_{i}_loss")


if __name__ == '__main__':
    print(f"-- Using {device} for training.")

    top_n_pred = [1, 2, 3]
    models = [load_m4]
    #datasets = ["2_cf_cr_optional", "3_cp_cf_cr_optional", "4_complete_without_return_expressions"]
    n_repetitions = 3

    #for dataset in datasets:
    dataset = "Complete"
    # Load data
    RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y = get_datapoints(dataset)
    #print(f"-- Loading data: {dataset}")
    Xr, yr = load_data_tensors(RETURN_DATAPOINTS_X, RETURN_DATAPOINTS_Y, limit=-1)
    Xp, yp = load_data_tensors(PARAM_DATAPOINTS_X, PARAM_DATAPOINTS_Y, limit=-1)
    X = torch.cat((Xp, Xr))
    y = torch.cat((yp, yr))

    # for load_model in models:
    #     for i in range(n_repetitions):
    #         model, model_config = load_model()
    #
    #         print(f"-- Model Loaded: {model} with {count_model_parameters(model)} parameters.")
    #
    #         train_loader, test_loader = load_dataset(X, y, model_config['batch_size'], split=0.8)
    #
    #         # Start training
    #         losses = train_loop(model, train_loader, model_config, model_store_dir=f"{load_model.__name__}/{dataset}/{i}"+str(int(time.time())))
    #
    #         # print("-- Loading model")
    #         # model = load_model('1571306801/model_BiRNN_e_9_l_1.8179169893.h5')
    #
    #         # Evaluate model performance
    #         y_true, y_pred = evaluate(model, test_loader, top_n=max(top_n_pred))
    #
    #         # If the prediction is "other" - ignore the result
    #         idx_of_other = pickle.load(open(f'./input_datasets/{dataset}/ml_inputs/label_encoder.pkl', 'rb')).transform(['other'])[0]
    #         idx = (y_true != idx_of_other) & (y_pred[:, 0] != idx_of_other)
    #
    #         for top_n in top_n_pred:
    #             filename = f"{load_model.__name__}_{dataset}_{i}_{top_n}"
    #             report(y_true, y_pred, top_n, filename)
    #             report(y_true[idx], y_pred[idx], top_n, f"{filename}_unfiltered")
    #
    #         report_loss(losses, f"{load_model.__name__}_{dataset}_{i}_loss")

# TypeWriter ################################################################################


class TWModel(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, aval_type_size: int, num_layers: int, output_size: int,
                 bidirectional=True):
        super(TWModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.aval_type_size = aval_type_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.lstm_id = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=self.bidirectional)
        self.lstm_tok = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                                bidirectional=self.bidirectional)
        self.lstm_cm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=self.bidirectional)

        self.linear = nn.Linear(hidden_size * 3 * (2 if self.bidirectional else 1) + self.aval_type_size, self.output_size)

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
        

def load_data_tensors_TW(filename, limit=-1):

    return torch.from_numpy(np.load(filename)[:limit]).float()


def load_label_tensors_TW(filename, limit=-1):

    return torch.from_numpy(np.argmax(np.load(filename), axis=1)[0:limit]).long()


def load_dataset_TW(X_id, X_tok, X_cm, X_type, Y, batch_size, split=0.7, workers=4):

    train_data = torch.utils.data.TensorDataset(X_id, X_tok, X_cm, X_type, Y)

    train_size = int(split * len(train_data))
    test_size = len(train_data) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_data, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


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


def report_TW(y_true, y_pred, top_n, total_dps, filename):

    y_pred_fixed = top_n_fix(y_true, y_pred, top_n)

    # pred_corr = np.count_nonzero(np.equal(y_true, y_pred_fixed))

    # precision = pred_corr / y_pred.shape[0]
    # recall = pred_corr / total_dps
    # f1_score = (2 * precision * recall) / (precision + recall)

    #print("p:", precision)
    #print("r:", recall)
    #print("f1:", f1_score)

    #report = {"weighted avg": {"precision": precision, "recall": recall, "f1-score": f1_score}}

    # Computation of metrics
    report = classification_report(y_true, y_pred_fixed, output_dict=True)
    print("Accuracy: ", report["macro avg"])
    store_json(report, f"{filename}.json", "./output/reports/json")

    return report

def train_loop_TW(model: nn.Module, data_loader, learning_rate, epochs):

    # Parameters
    #learning_rate = 0.002
    #epochs = 2

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(data_loader)
    losses = np.empty(total_step * epochs)

    i = 0
    for epoch in range(1, epochs + 1):
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
            losses[i] = loss.item()
            i += 1
            print(f'Epoch [{epoch}/{epochs}], Batch: [{batch_i}/{total_step}], '
                  f'Loss:{loss.item():.10f}')


def main_TW():

    # Parameters
    input_size = 14
    hidden_size = 10
    output_size = 1000
    num_layers = 1
    top_n_pred = [1, 2, 3]
    n_rep = 3

    # Loading parameters' data vectors
    X_id_param = load_data_tensors_TW('./output/vectors/TW/identifiersparam_datapoints_x.npy')
    X_tok_param = load_data_tensors_TW('./output/vectors/TW/tokensparam_datapoints_x.npy')
    X_cm_param = load_data_tensors_TW('./output/vectors/TW/commentsparam_datapoints_x.npy')

    # Loading return' data vectors
    X_id_ret = load_data_tensors_TW('./output/vectors/TW/identifiersret_datapoints_x.npy')
    X_tok_ret = load_data_tensors_TW('./output/vectors/TW/tokensret_datapoints_x.npy')
    X_cm_ret = load_data_tensors_TW('./output/vectors/TW/commentsret_datapoints_x.npy')

    # Loading labels
    Y_param = load_label_tensors_TW('./output/vectors/TW/params_datapoints_y.npy')
    Y_ret = load_label_tensors_TW('./output/vectors/TW/returns_datapoints_y.npy')


    X_id = torch.cat((X_id_param, X_id_ret))
    X_tok = torch.cat((X_tok_param, X_tok_ret))
    X_cm = torch.cat((X_cm_param, X_cm_ret))

    Y = torch.cat((Y_param, Y_ret))

    train_loader, test_loader = load_dataset_TW(X_id, X_tok, X_cm, Y, 128)

    model = TWModel(input_size, hidden_size, num_layers, output_size, True).to(device)
    print(f"-- Using {device} for training.")

    for i in range(1, n_rep+1):

        train_loop_TW(model, train_loader)

        # Evaluate model performance
        y_true, y_pred = evaluate_TW(model, test_loader, top_n=max(top_n_pred))

        # TODO: Ignore other or unknown type

        for top_n in top_n_pred:
            # TODO: Add report for unfiltered results
            filename = f"{TWModel.__name__}_complete_{i}_{top_n}_unfiltered"
            report_TW(y_true, y_pred, top_n, filename)

#############################################################################################

