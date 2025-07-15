import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from torch.utils.data import DataLoader
from dataset import ViT_HER2ST, HER2ST, UNI_HER2ST, WSUNI_HER2ST, WS_UNI_HER2ST
from models.HisToGene_model import HisToGene
from models.STNet_model import STModel
from models.UNI import UNI
from models.WSUNI import WSUNI
from models.WS_UNI import WS_UNI
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Login W&B
wandb.login(key="c0bf463d253eb9147fbe555216398f2838fe517c")
wandb_logger = WandbLogger(project="ST", entity="dacthai2807", name="WSUNI_adapk_pos_50epoch")

fold = 5
tag = '-htg_her2st_785_32_cv'

mode = "WSUNI"#input("Choose model to train [Histogene/ST-Net/UNI/WS_UNI]: ")

if mode == "Histogene":
    dataset = ViT_HER2ST(train=True, fold=fold)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    model = HisToGene(n_layers=8, n_genes=785, learning_rate=1e-5)

    trainer = pl.Trainer(accelerator='gpu', max_epochs=100)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint("model_ckpts/histogene_last_train_"+tag+'_'+str(fold)+".ckpt")

elif mode == "UNI":
    checkpoint_callback = ModelCheckpoint(
        dirpath='model_ckpts/UNI_final',
        filename='UNI_every5epoch_'+tag+'_'+str(fold)+'_{epoch}',
        save_top_k=-1,
        every_n_epochs=5,
        save_last=True
    )

    dataset = UNI_HER2ST(train=True, fold=fold)
    train_loader = DataLoader(dataset, batch_size=16, num_workers=1, shuffle=True)

    model = UNI(
        n_genes=785,
        learning_rate=1e-5,
        max_epochs=50
    )
    model.enable_lora_training()

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=50,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )
    trainer.fit(model, train_loader)
    
elif mode == "WSUNI":
    checkpoint_callback = ModelCheckpoint(
        dirpath='model_ckpts/WSUNI_adapk_pos_50epoch',
        filename='WSUNI_every5epoch_'+tag+'_'+str(fold)+'_{epoch}',
        save_top_k=-1,
        every_n_epochs=5,
        save_last=True
    )

    dataset = WSUNI_HER2ST(train=True, fold=fold, cache_dir='cache_features_train/', topk=40)
    train_loader = DataLoader(dataset, batch_size=16, num_workers=1, shuffle=True)

    model = WSUNI(
        n_genes=785,
        learning_rate=1e-5,
        max_epochs=50
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=50,
        callbacks=[checkpoint_callback],
        logger=wandb_logger
    )
    trainer.fit(model, train_loader)

elif mode == "WS_UNI":
    checkpoint_callback = ModelCheckpoint(
        dirpath='model_ckpts/WS_UNI',
        filename='WS_UNI_every5epoch_'+tag+'_'+str(fold)+'_{epoch}',
        save_top_k=-1,
        every_n_epochs=5,
        save_last=True
    )

    dataset = WS_UNI_HER2ST(train=True, fold=fold)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)

    model = WS_UNI(
        uni_ckpt_path="model_ckpts/UNI_v2/UNI_every5epoch_-htg_her2st_785_32_cv_5_epoch=44.ckpt",
        n_layers=6,
        n_genes=785,
        learning_rate=1e-5,
        max_epochs=50
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=50,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=1
    )
    trainer.fit(model, train_loader)

elif mode == "ST-Net":
    dataset = HER2ST(train=True, fold=fold)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    model = STModel(n_genes=785, learning_rate=1e-5)

    trainer = pl.Trainer(accelerator='gpu', max_epochs=100)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint("model_ckpts/stnet_last_train_"+tag+'_'+str(fold)+".ckpt")

else:
    print("error")
