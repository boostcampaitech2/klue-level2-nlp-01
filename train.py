import re
import os

from pytorch_lightning.callbacks import early_stopping
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import Bertbase

def main():
    klue_data = DataModule("monologg/koelectra-base-v3-discriminator")
    model = Bertbase("monologg/koelectra-base-v3-discriminator")

    num_duplicated_model_dir = len([re.search(model._get_name(), dir) for dir in os.listdir("./models")])
    dirpath = f"./models/{model._get_name()}{num_duplicated_model_dir}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath, monitor="val_loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir="logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=20,
        precision=16,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("logs/", name="cola", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    trainer.fit(model, klue_data)

if __name__ == "__main__":
    main()
