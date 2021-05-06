# Copyright (c) 2020 Hanwei Wu
# based on the paper 
# Conditional Mutual information-based Contrastive Loss for Financial Time Series Forecasting (Hanwei Wu, Ather Gattami, Markus Flierl)
import torch.nn as nn
num_samples = 2
class contrative_loss(nn.Module):
    def __init__(self, temp):
        super().__init__()
        #self.bilinear = nn.Bilinear(latent_dim, latent_dim, 1)
        self.temp = temp
        
    def forward(self, x, y):
        """
        @type  x: tensor
        @type  y: tensor
        @rtype    float
        """
        losses = []
        bs_len = len(x)  #batch_size
        samples = torch.randint(0, bs_len, (bs_len,num_samples)) # sample size (bs_len, num_samples)
        for i in range(bs_len):
            sim_scores = torch.einsum('ij,ij->i', x[i].repeat(num_samples,1),x[samples[i]])/self.temp
            diff_abs = torch.abs(sim_scores[0]-sim_scores[1])
            if y[samples[i][0]] == y[samples[i][1]]:
              label = 0
            else:
              label = 1
            
            # inner product between [x[i], x[i]] with [x[i], sample] 
            sample_pair = torch.stack((x[i], x[samples[i]].squeeze()),0) 
            sim_scores = torch.einsum('ij,ij->i', x[i].repeat(2,1),sample_pair)
            diff_abs = (sim_scores[0]-sim_scores[1])/temp
            
            pred_prob = 1/(1+torch.exp(-diff_abs))

           # sim_scores = torch.dot(x[i].repeat(num_samples, 1),x[samples[i]].squeeze(0))
           # pred_prob = 1/(1+torch.exp(-sim_scores))
            
            loss_term1 = -torch.log(pred_prob+1e-5)*label
            loss_term2 = - torch.log(1-pred_prob+1e-5)*(1-label)
            loss = loss_term1 + loss_term2
            losses.append(loss)
        return torch.mean(torch.stack(losses))


class binary_constrastive(nn.Module):
  def __init__(self, temp):
    super().__init__()
    self.temp = temp
        
  def forward(self, x, y):
    """
    @type  x: tensor
    @type  y: tensor
    @rtype    float
    """
    losses = []
    bs_len = len(x)  #batch_size
    samples = torch.randint(0, bs_len, (bs_len,1)) # sample size (bs_len, num_samples)
    for i in range(bs_len):
        sample_pair = torch.stack((x[i], x[samples[i]].squeeze()),0) 
        sim_scores = torch.einsum('ij,ij->i', x[i].repeat(2,1),sample_pair)/temp
        diff_abs = sim_scores[0]-sim_scores[1]
        if y[i] == y[samples[i]]:
          label = 0
        else:
          label = 1        
        pred_prob = 1/(1+torch.exp(-diff_abs))
        
        loss_term1 = - torch.log(pred_prob+1e-5)*label
        loss_term2 = - torch.log(1-pred_prob+1e-5)*(1-label)
        loss = loss_term1 + loss_term2
        losses.append(loss)
    return torch.mean(torch.stack(losses))