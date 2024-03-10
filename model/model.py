import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ndcg_score


class RankingModel(pl.LightningModule):
    def __init__(self, input_size, learning_rate):
        super().__init__()
        self.lr = learning_rate

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.loss_fn = F.binary_cross_entropy

        self.validation_scores_targets = []
        self.test_scores_targets = []

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

    def _score_ndcg(self, scores_targets, log_name: str):
        all_targets = torch.cat([item[0] for item in scores_targets]).cpu().numpy()
        all_scores = torch.cat([item[1] for item in scores_targets]).cpu().numpy()
        ndcg = ndcg_score(all_targets.reshape(1, -1), all_scores.reshape(1, -1))
        self.log(log_name, ndcg)
        scores_targets.clear()

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._sharable_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._sharable_step(batch, batch_idx)
        self.validation_scores_targets.append((y, scores))
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        self._score_ndcg(self.validation_scores_targets, "val_ndcg")

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._sharable_step(batch, batch_idx)
        self.test_scores_targets.append((y, scores))
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self):
        self._score_ndcg(self.test_scores_targets, "test_ndcg")

    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs).squeeze(1)
        preds = torch.argmax(outputs, dim=1)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
