# Exercise set 5

*All assignments need to be implemented within the function skeletons found in `submission.py`
and you need to hand in this file in the form `submission_<STUDENTID>.py` at the link provided
for this exercise sheet via e-mail.*

### Exercise 5.1

In this exercise, we train a simple MLP for *fuel economy* prediction of vehicles. The raw data contains 13 features per vehicle and the *fuel economy* per vehicle as our target variable.

We have 1107 vehicles available. The data loading and preprocessing code is provided. 
Once the dataset is transformed (by the provided code), we have 50 features per vehicle, so our data tensor is of shape (1107,50). **Important**: our target variable (i.e., the *fuel economy*) is real-valued *and* log-transformed. This means that once training is done and 
we want to compute, e.g., the mean-squared-error (MSE) of the training predictions and the true ground truth, we need to exponentiate our predictions.

In the provided template code (see function `train`), you find the lines

```python
ds_trn, _ = get_data()
dl_trn = DataLoader(ds_trn, batch_size=32, shuffle=True)
```

These two lines load the data into a PyTorch dataset and create a PyTorch dataloader (providing batches of size 32). As we have seen in the lecture, we can iterate over the 
dataloader using 

```python
for x_trn, y_trn in dl_trn:
    # do something
```

Your task is to (1) fill up the `MLP` class defining an appropriate MLP (for the task) with ***at most** 1000 parameters in total (e.g., a linear layer followed by a ReLU and another linear layer would be a good starting point) and then (2) adjust the `train` method accordingly to minimize the MSE down to an error (computed over the full datasets) of less than 3.0. The grader will check for an error of <3.0 and whether the restriction on the number of parameters is satisfied.