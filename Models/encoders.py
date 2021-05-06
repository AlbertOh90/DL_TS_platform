#Copyright (c) 2020 Hanwei Wu
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
#########################
## CNN-based encoders ##
########################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

###################################
## Autoregressive-based encoders ##
###################################

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
      

###################################
####   Enocders for inference  ####
###################################
def feature_extracter(model, device, x_train, x_valid, x_test):
  model.eval()
  with torch.no_grad():
    test_codes = model.encoder(x_test.to(device)).squeeze()
    test_labels = y_test
    train_codes = model.encoder(x_train.to(device)).squeeze()
    train_labels = y_train
    valid_codes = model.encoder(x_valid.to(device)).squeeze()
    valid_lables = y_valid
  return train_codes, valid_codes, test_codes 

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, features):
        return self.fc(features)

