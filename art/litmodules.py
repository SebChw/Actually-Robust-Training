import pytorch_lightning as pl
import torch
import torch.nn.functional as F



class LitAudioClassifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def processing_step(self, batch, prompt):
        x, y = batch["data"], batch["label"]
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log(prompt, loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.processing_step(batch, "train_loss")
    
    def validation_step(self, batch, batch_idx):
        self.processing_step(batch, "val_loss")
    
    def test_step(self, batch, batch_idx):
        self.processing_step(batch, "test_loss")
       
            