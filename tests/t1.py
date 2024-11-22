from otter.test_files import test_case

OK_FORMAT = False

from torch.utils.data import DataLoader
from utils import get_data

name = "Exercise 5.1"
points = 4

@test_case(points=3)
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