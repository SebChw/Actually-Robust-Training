import lightning.pytorch as pl


class ClassificationModel(pl.LightningModule):
    def __init__(self, model):
        self.model = model

    # We define structure here for typical classification, one just needs to add their own module.
    # TODO we should define templates for tasks, start from classification.
