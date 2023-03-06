import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


class LitAudioClassifier(pl.LightningModule):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def processing_step(self, batch, prompt):
        x, y = batch["data"], batch["label"]
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log(f"{prompt}_loss", loss)

        self.accuracy(logits, y)
        self.log(f'{prompt}_acc', self.accuracy, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.processing_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        self.processing_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        self.processing_step(batch, "test")
       
            