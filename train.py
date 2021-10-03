import os
import time
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data import DataModule
from model import ModelModule
from utils import increment_path

def main(args):

    start = time.time()
    dirpath = increment_path(f"./models/{args.model_name}")
    os.makedirs(dirpath, exist_ok=True)
    
    # 이후에 argument 전달과 parsing 구조 바꾸기
    klue_data = DataModule(args.model_name, batch_size=args.batch_size, version=args.version)
    model = ModelModule(args.model_name, lr=args.lr, dirpath=dirpath, version=args.version)

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath, monitor="eval_micro f1 score", mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # pl.seed_everything(42, workers=True)
    trainer = pl.Trainer(
        max_epochs=20,
        gpus=-1,
        precision=16,
        accumulate_grad_batches=4,
        log_every_n_steps=20,
        val_check_interval=0.5,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    trainer.fit(model, klue_data)
    print(f"total training time ... {(time.time()-start)//60} minutes ...")

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="klue/bert-base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--version", type=int, help="1: origin , 2: type-entity-marker", default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    
    main(args)
