#Copyright (c) 2020 Hanwei Wu
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
from statsmodels.tsa.arima_model import ARIMA

def wMAPA(true_d, pred_d):
  num = 0 
  denum = 0
  for i, j in zip(true_d, pred_d):
    num += abs(i-j)
    denum += i
  return 1-num/denum
  
def mse(x, y):
  return torch.nn.MSELoss()(x, y)


class NNregressor():
  def __init__(self, latent_dim, dilation_depth, attention_step, normalization):
    self.model = ARautoencoder(1, latent_dim, dilation_depth, attention_step, normalization)

  def train(self, train_hist, train_tar, bs, save_dir, save_model = True):
    train_ds = TensorDataset(data_standlization(torch.FloatTensor(train_hist)), data_standlization(torch.FloatTensor(train_tar)))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    optimizer = optim.Adam(self.model.parameters())
    for epoch in range(epochs):
        self.model.train()
        loss_accu = []
        for xb, yb in train_dl:
            optimizer.zero_grad()
            value_vec = self.model(xb.unsqueeze(1))
            loss = mse(value_vec.squeeze(), yb)
            loss_accu.append(loss)
            loss.backward()
            optimizer.step()
        loss_print = torch.mean(torch.stack(loss_accu))
        print('epoch:',epoch,'training loss:',loss_print)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': self.model.state_dict(), 
        'optimizer' : optimizer.state_dict()}, False, save_dir)
    #return self.model
  
  def forecast(self, test_hist):
    test_hist, means, stds = standlization(torch.FloatTensor(test_hist))
    self.model.eval()
    with torch.no_grad():
      preds = self.model(test_hist.unsqueeze(1))
      #train_preds = model(torch.FloatTensor(train_hist).unsqueeze(1))
    return destandlization(preds, means, stds)
  
  def load_model(self, save_dir):
    checkpoint = torch.load(os.path.join(save_dir,'checkpoint.pth.tar'))
    self.model.load_state_dict(checkpoint['state_dict'])


def ARIMA_model(train_set, test_set, forca_lag):
  history = [x for x in train_set]
  preds = []
  for t in range(len(test_set)):
    model = ARIMA(history, order=(4,0,1))
    model_fit = model.fit(disp=0)
    yhat  = model_fit.forecast(steps=forca_lag)[0][-1]
    preds.append(yhat)
    obs = test_set[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
  print('wMAPA is: %.3f' % wMAPA(test_set, preds))
  return preds