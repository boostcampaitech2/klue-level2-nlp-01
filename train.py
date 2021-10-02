import re
import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, lr_monitor
from transformers.utils.dummy_pt_objects import Trainer

from data import DataModule
from model import ModelModule

def main(args):

    num_duplicated_model_dir = len([re.search(args.model_name, d) for d in os.listdir("./models")])
    dirpath = f"./models/{args.model_name}{num_duplicated_model_dir}"

    klue_data = DataModule(args.model_name, batch_size=args.batch_size, version=args.version)
    model = ModelModule(args.model_name, lr=args.lr, dirpath=dirpath)

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath, monitor="eval_micro f1 score", mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # pl.seed_everything(42, workers=True)
    trainer = pl.Trainer(
        max_epochs=20,
        precision=16,
        accumulate_grad_batches=4,
        log_every_n_steps=20,
        val_check_interval=20,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, klue_data)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="klue/bert-base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--vesion", type=int, help="1: origin , 2: type-entity-marker", default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    
    main(args)
