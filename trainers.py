import torch
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from TS_platform.util import save_checkpoint

class Trainer:
    """
    args:
    model：pytorch nn.module
    train_args:  training args using @dataset
    train_dataset: pytorch Dataset
    eval_dataset: pytorch Dataset
    """

    def __init__(self, model=None, train_args=None, train_dataset=None, eval_dataset=None, loss=None, device = None):
        self.model = model  # the instantiated model to be trained
        self.args = train_args  # training arguments
        self.train_dataset = train_dataset  # training dataset
        self.eval_dataset = eval_dataset  # evaluation dataset
        self.loss = loss
        self.optimizer = optim.Adam(self.model.parameters())
        self.best_val_loss = None
        self.device = device

    def train(self, save_model=True):
        train_accs = []
        test_accs = []
        train_dl = DataLoader(self.train_dataset, batch_size=self.args.bs, shuffle=True)
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(self.args.epochs):
            self.fit()
            val_loss = self.evaluate()
            if not self.best_val_loss or val_loss <= self.best_val_loss:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, self.args.save_dir, epoch)
                self.best_val_loss = val_loss

            print('epoch:', epoch, 'training loss:', val_loss)
            self.model.eval()
            with torch.no_grad():
                train_acc = accuracy_score(torch.argmax(self.model(self.train_dataset[:][0].to(self.device)).cpu(), axis=1),
                                           self.train_dataset[:][1])
                test_acc = accuracy_score(torch.argmax(self.model(self.eval_dataset[:][0].to(self.device)).cpu(), axis=1),
                                          self.eval_dataset[:][1])
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        return train_accs, test_accs

    def evaluate(self):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        total_loss = 0.
        val_loader = DataLoader(dataset=self.eval_dataset, batch_size=self.args.eval_batch_size)
        for step, batch in enumerate(val_loader):
            with torch.no_grad():
                batch = tuple(t.to(self.device) for t in batch)
                xb, target_b = batch
                output = self.model(xb)
                loss = self.loss(output, target_b).item()
                total_loss += loss
        return total_loss / (step + 1)

    def fit(self):
        total_loss = 0
        self.model.train()
        start_time = time.time()
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.bs, shuffle=True)

        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(self.device) for t in batch)
            xb, target_b = batch
            output = self.model(xb)
            loss = self.loss(output, target_b)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            """
            total_loss += loss.item()
            if step % self.args.log_interval == 0 and step > 0:
                cur_loss = total_loss / (self.args.log_interval + 1)
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} steps | lr {:02.4f} | ms/step {:5.2f} | '
                          'loss {:5.2f} | ppl {:8.2f}'.format(
                        self.epoch, step, len(train_dataset) // self.args.bs, lr,
                                      elapsed * 1000 / (self.args.log_interval + 1), cur_loss, math.exp(cur_loss)))
                total_loss = 0
            """