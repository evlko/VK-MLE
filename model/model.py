import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import ndcg_score


class RankingModel(pl.LightningModule):
    def __init__(self, input_size, learning_rate):
        super().__init__()
        self.lr = learning_rate

        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.loss_fn = F.binary_cross_entropy

        self.validation_step_targets = []
        self.validation_step_outputs = []

        self.test_step_targets = []
        self.test_step_outputs = []

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def _sharable_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs).squeeze(1)
        loss = self.loss_fn(outputs, targets.float())
        return loss, outputs, targets

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._sharable_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._sharable_step(batch, batch_idx)
        self.validation_step_targets.append(y)
        self.validation_step_outputs.append(scores)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        all_preds = (
            torch.cat([out for out in self.validation_step_outputs]).cpu().numpy()
        )
        all_targets = (
            torch.cat([out for out in self.validation_step_targets]).cpu().numpy()
        )
        ndcg = ndcg_score(all_targets.reshape(1, -1), all_preds.reshape(1, -1))
        self.log("val_ndcg", ndcg)
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._sharable_step(batch, batch_idx)
        self.test_step_targets.append(y)
        self.test_step_outputs.append(scores)
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self):
        all_preds = torch.cat([out for out in self.test_step_outputs]).cpu().numpy()
        all_targets = torch.cat([out for out in self.test_step_targets]).cpu().numpy()
        ndcg = ndcg_score(all_targets.reshape(1, -1), all_preds.reshape(1, -1))
        self.log("test_ndcg", ndcg)
        self.test_step_outputs.clear()
        self.test_step_targets.clear()

    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs).squeeze(1)
        preds = torch.argmax(outputs, dim=1)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
