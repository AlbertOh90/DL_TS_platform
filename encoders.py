import torch
import torch.nn as nn
import torch.nn.functional as F

class ARautoencoder(nn.Module):
  def __init__(self, num_features, latent_dim, dilation_depth, attention_step, num_targets, normalization):
    super().__init__()
    self.auto = Autoregressive(num_features, latent_dim, 1, dilation_depth)
    self.attention = self_attention(attention_step, latent_dim)
    self.encoder = nn.Sequential(self.auto, self.attention)
    self.head = Projecter(latent_dim, num_targets, normalization)
  def forward(self, x):
    x = self.encoder(x)
    #x = x.squeeze()
    x = self.head(x)
    return x

# reference: https://lirnli.wordpress.com/2017/10/16/pytorch-wavenet/
class Autoregressive(nn.Module):
    def __init__(self, num_features, latent_dim, contratvie_dim, dilation_depth, n_residue=128):
        """
        n_residue: residue channels
        latent_dim skip channels
        dilation_depth: dilation layer setup
        """
        super().__init__()
        self.dilation_depth = dilation_depth
        dilations = self.dilations = [2 ** i for i in range(dilation_depth)]
        self.from_input = nn.Conv1d(in_channels=num_features, out_channels=n_residue, kernel_size=1)
        self.conv_sigmoid = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
             for d in dilations])
        self.conv_tanh = nn.ModuleList(
            [nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=2, dilation=d)
             for d in dilations])
        self.skip_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=latent_dim, kernel_size=1)
                                         for d in dilations])
        self.residue_scale = nn.ModuleList([nn.Conv1d(in_channels=n_residue, out_channels=n_residue, kernel_size=1)
                                            for d in dilations])

    def forward(self, inputs):
        output = self.preprocess(inputs)
        skip_connections = []  # save for generation purposes
        for s, t, skip_scale, residue_scale in zip(self.conv_sigmoid, self.conv_tanh, self.skip_scale,
                                                   self.residue_scale):
            output, skip = self.residue_forward(output, s, t, skip_scale, residue_scale)
            skip_connections.append(skip)
        # sum up skip connections
        output = sum([s[:, :, -output.size(2):] for s in skip_connections])
        return output

    def preprocess(self, inputs):
        # increase the channel numbers
        output = self.from_input(inputs)
        return output

    def residue_forward(self, inputs, conv_sigmoid, conv_tanh, skip_scale, residue_scale):
        output = inputs
        output_sigmoid, output_tanh = conv_sigmoid(output), conv_tanh(output)
        output = torch.sigmoid(output_sigmoid) * torch.tanh(output_tanh)
        skip = skip_scale(output)
        output = residue_scale(output)
        output = output + inputs[:, :, -output.size(2):]
        return output, skip

class Projecter(nn.Module):
    def __init__(self, input_dim, output_dim, norm = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = norm
    def forward(self, x):
        x = self.linear(x)
        if self.norm:
          return F.normalize(x, dim = 1)
        else:
          return x

class self_attention(torch.nn.Module):
  def __init__(self, num_timesteps, latent_dim):
    super().__init__()
    self.weights = nn.Sequential(nn.Linear(latent_dim, 1), nn.Softmax(dim = 1))
    self.num_timesteps = num_timesteps

  def forward(self, z):
      """
      args:
      z  -- tensors of shape (batch_num, step_num, latent_dim)
      """
      z = z.permute(0,2,1) #(batch_num, step_num, latent_dim)
      z = z[:,-self.num_timesteps:,:]
      weights = self.weights(z)   #shape:(batch_num, step_num, 1)
      return torch.sum(weights*z, dim=1)  #shape: (batch_num, latent_num)
