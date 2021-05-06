import torch
from TS_platform.util import data_standlization
from TS_platform.encoders import ARautoencoder
from TS_platform.trainers import Trainer

class Classifer_Autoregressive:
    def __init__(self, num_features, latent_dim, dilation_depth, attention_step, num_targets, normalization, device):
        self.model = ARautoencoder(num_features, latent_dim, dilation_depth, attention_step, num_targets,
                                   normalization).to(device)
        self.best_val_loss = None
        self.device = device

    def train(self, train_args=None, train_dataset=None, eval_dataset=None, loss=None):
        trainer = Trainer(model=self.model, train_args=train_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                          loss=loss)
        return trainer.train()

    def predict(self, test_hist):
        self.model.eval()
        with torch.no_grad():
            pred_logits = self.model(data_standlization(test_hist))
            # train_preds = model(torch.FloatTensor(train_hist).unsqueeze(1))
        return torch.argmax(pred_logits, axis=1)

    def load_model(self, save_dir):
        checkpoint = torch.load(save_dir, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])