import torch
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim
import lightning as lt
from utils import Parser as P

class NN(lt.LightningModule):
    def __init__(self,
                 config_file: str = 'vae_config.yaml',
                 **kwargs) -> None:
        super(NN, self).__init__()
        self.save_hyperparameters()

        # Read model file and build networks
        self.config = P.read_config(config_file)

        self.classifier = P.network(self.config['classifier'])
        self.loss_fun = P.loss_fun(self.config['loss_fun'])

        # Read more: https://lightning.ai/docs/torchmetrics/stable/all-metrics.html
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['num_classes'])
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['num_classes'])
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.config['num_classes'])

    def forward(self, data):
        pred = self.classifier(data)
        return pred

    def loss_function(self, pred, label):
        return self.loss_fun(pred, label)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.005)

    def training_step(self, batch, batch_idx):
        data, label = batch
        pred = self(data)
        train_loss = self.loss_function(pred, label)
        self.train_acc(pred, label)
        values = {"train_loss": train_loss, "train_acc": self.train_acc}
        self.log_dict(values, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        pred = self(data)
        val_loss = self.loss_function(pred, label)
        self.val_acc(pred, label)
        values = {"val_loss": val_loss, "val_acc": self.val_acc}
        self.log_dict(values, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, label = batch
        pred = self(data)
        test_loss = self.loss_function(pred, label)
        self.test_acc(pred, label)
        values = {"test_loss": test_loss, "test_acc": self.test_acc}
        self.log_dict(values, prog_bar=True)

