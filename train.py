import argparse
import logging
import os
import string

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from model import InductionTransformer
from datasets import get_dataset



def main(
    lr,
    weight_decay,
    beta1,
    beta2,
    heads,
    layers,
    width,
    seq_first,
    dropout,
    context_len,
    num_samples,
    data_dir,
    force_data,
    batch_size,
    steps,
    train_ratio,
    seed,
    verbose,
    log_freq,
    num_workers,
    disable_logging,
    checkpoint_every,
):
    # set wandb logging mode
    if disable_logging:
        os.environ['WANDB_MODE'] = 'offline'
    else:
        os.environ['WANDB_MODE'] = 'online'
    
    # set logging level
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # seeding
    if seed is not None:
        pl.seed_everything(seed)

    # data
    data = get_dataset(context_len=context_len, num_samples=num_samples, data_dir=data_dir, force_data=force_data)
    idcs = np.random.permutation(np.arange(len(data)))
    train_idcs = idcs[:int(train_ratio * len(idcs))]
    val_idcs = idcs[int(train_ratio * len(idcs)):]
    if batch_size == -1:
        train_batch_size = len(train_idcs)
        val_batch_size = len(val_idcs)
    else:
        train_batch_size = batch_size
        val_batch_size = batch_size
    train_loader = DataLoader(Subset(data, train_idcs), batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(data, val_idcs), batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # model
    optim_kwargs = {
        'lr': lr,
        'weight_decay':weight_decay,
        'betas': (beta1, beta2),
    }
    num_tokens = len(string.ascii_lowercase)
    model_kwargs = {
        'heads':heads,
        'layers':layers,
        'width':width,
        'num_tokens':num_tokens,
        'optim_kwargs':optim_kwargs,
        'dropout':dropout,
        'batch_first':not seq_first,
    }
    model = InductionTransformer(**model_kwargs)

    # wandb config
    config = dict(
        **optim_kwargs, 
        **model_kwargs, 
        batch_size=batch_size, 
        steps=steps, 
        seed=seed,
        context_len=context_len,
        num_samples=num_samples,
    )

    # callbacks
    callbacks = []
    callbacks.append(
        pl.callbacks.model_checkpoint.ModelCheckpoint(
            verbose=verbose,
            monitor="Validation/Accuracy",
            auto_insert_metric_name=True,
            every_n_epochs=checkpoint_every,
            save_top_k=-1,
        )
    )
    callbacks.append(pl.callbacks.progress.TQDMProgressBar(refresh_rate=log_freq))
    callbacks.append(pl.callbacks.lr_monitor.LearningRateMonitor())
    
    # training
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_steps=steps,
        log_every_n_steps=log_freq,
        callbacks=callbacks,
        logger=WandbLogger(project="mvp_induction", config=config),
    )
    trainer.fit(model, train_loader, val_loader)
    
    callbacks[0].to_yaml()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # optimizer args
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)

    # model args
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--seq_first", action="store_true", help="Whether to have time dim first or batch dim first")
    

    # data args
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--force_data", action="store_true", help="Whether to force dataset creation.")
    
    # training args
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10**5)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_logging", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=0, help="Save a checkpoint every n epochs")
    
    # collect args and run main
    args = parser.parse_args()
    if args.verbose:
        print(f'Arguments = {args}')

    main(**vars(args))
