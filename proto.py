from data_utils import data_loader, class_weights

from argparse import ArgumentParser

from model import Layoutlmv3Model, ViltModel

import pytorch_lightning as L

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import wandb

import json

###############################################################################
# Runtime parameters ##########################################################
###############################################################################
arg_parser = ArgumentParser(description='Run an experiment.')
arg_parser.add_argument('--dataset', type=str, required=True,
                         help='Pick a dataset {memotion7k, multioff, fbhm}',
                         choices={'memotion7k', 'multioff', 'fbhm'})
arg_parser.add_argument('--batch', '-b', type=int, default=64,
                         help='Batch size',)
arg_parser.add_argument('--freeze', '-f', type=int, default=0,
                         help='Frozen layers',)

args = arg_parser.parse_args()

DATA_CODE = args.dataset
MODEL_CODE = 'dandelin/vilt-b32-finetuned-nlvr2'
BATCH_SIZE = 64 if args.batch > 64 else args.batch
ACCUM_STEPS = 1 if args.batch <= 64 else args.batch//64
TRAIN_EPOCHS = 20 * (128//BATCH_SIZE*ACCUM_STEPS) #1000 * (512/BATCH_SIZE*ACCUM_STEPS)  #1000 # 1000 for Memotion7k 500 for multioff
FROZEN = args.freeze 
LR = (BATCH_SIZE*ACCUM_STEPS/64)* 1e-5 # Learning rate proportional to batch size (64 == 3e-5)

###############################################################################
# MAIN ########################################################################
###############################################################################
def main():
    train = data_loader(DATA_CODE, 'train', MODEL_CODE, bs=BATCH_SIZE)
    dev = data_loader(DATA_CODE, 'dev', MODEL_CODE, bs=BATCH_SIZE*4)
    #test = data_loader(CODE, 'test', 'dandelin/vilt-b32-finetuned-vqa', bs=BATCH_SIZE)

    architecture = ViltModel
    model = architecture(MODEL_CODE, 
                        class_weights(DATA_CODE),
                        training_steps=TRAIN_EPOCHS*len(train)//ACCUM_STEPS,
                        lr=LR,
                        frozen=FROZEN)
    
    # Callbacks
    wandb.login()
    wandb_logger = WandbLogger(project = "multimodal_image_analysis")
    wandb_logger.experiment.config.update({'Dataset':DATA_CODE,
                                           'Batch size': BATCH_SIZE*ACCUM_STEPS, 
                                           'Frozen layers': FROZEN,
                                           'Learning rate': LR,
                                           'Scheduling': f'Linear decay',
                                           'Train epochs': TRAIN_EPOCHS,
                                           'Backend model': MODEL_CODE,
                                           })
    callbacks = [
          ModelCheckpoint(
            verbose=True,
            dirpath = f"checkpoints/{DATA_CODE}",
            every_n_epochs= 5 if DATA_CODE == 'multioff' else 1,
            save_top_k=1,
            monitor='valid_mean_f1'
        ), 
    ]

    trainer = L.Trainer(max_epochs=TRAIN_EPOCHS, 
                        devices=[0], 
                        precision=16, 
                        callbacks=callbacks,
                        check_val_every_n_epoch= 5 if DATA_CODE == 'multioff' else 1,
                        logger=wandb_logger,
                        accelerator='gpu',
                        log_every_n_steps=10,
                        accumulate_grad_batches=ACCUM_STEPS,
                        )
    
    trainer.fit(model, train, dev)
    results = trainer.test(model, dev, ckpt_path=callbacks[0].best_model_path)
    with open(f"results/{wandb_logger.experiment.name}.json", 'w') as f:
        json.dump(results, f)

main()