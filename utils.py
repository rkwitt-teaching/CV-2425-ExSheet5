"""utils.py - Data loading helpers - DO NOT MODIFY!"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader,TensorDataset


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