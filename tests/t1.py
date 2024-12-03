from otter.test_files import test_case

OK_FORMAT = False

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader,TensorDataset

name = "Exercise 5.1"
points = 6


def get_data():
    # Load data
    fuel = pd.read_json('assets/fuel.json')
    X = fuel.copy()
    y = X.pop('FE')

    # Preprocess data
    preprocessor = make_column_transformer(
        (StandardScaler(),
        make_column_selector(dtype_include=np.number)),
        (OneHotEncoder(sparse_output=False),
        make_column_selector(dtype_include=object)),
    )
    X = preprocessor.fit_transform(X)
    y = np.log(y)
    
    # Split data into training/testing
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Create PyTorch datasets
    ds_trn = TensorDataset(
        torch.tensor(np.array(X_trn), dtype=torch.float32),
        torch.tensor(np.array(y_trn), dtype=torch.float32))
    ds_tst = TensorDataset(
        torch.tensor(np.array(X_tst), dtype=torch.float32),
        torch.tensor(np.array(y_tst), dtype=torch.float32))
    return ds_trn, ds_tst


@test_case(points=4)
def test_1(train, env):
    _, ds_tst = get_data()
    dl_tst = DataLoader(ds_tst, batch_size=32, shuffle=False)
    
    model = env['train']()
    model.eval()
    
    N = 0
    error = 0
    for x_batch, y_batch in dl_tst:
        y_pred = model(x_batch.float())
        error += (y_pred.view(-1).exp()-y_batch.view(-1).exp()).abs().sum().item() 
        N += y_batch.shape[0]
    
    assert error/N < 3.0, "Error too high: {}".format(error/N)


@test_case(points=2)
def test_2(train, env):
    model = env['train']()
    num_params = np.sum([p.numel() for p in model.parameters()])
    assert num_params < 1000, "Too many parameters: {}".format(num_params)