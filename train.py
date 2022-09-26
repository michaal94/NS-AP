import os
import sys
import json
import argparse
from model import supervisor
from model.supervisor import Supervisor
from config import get_config
from dataset import get_dataloader
from model import get_trainer, get_supervisor


PROJECT_PATH = os.path.abspath('.')
sys.path.insert(0, PROJECT_PATH)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--stage', default='segmentation'
)
parser.add_argument(
    '--config', default=''
)

def main(args):
    if args.config == '':
        cfg = get_config(args.stage)
    else:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    
    train_loader = get_dataloader(cfg['dataset_train'])
    val_loader = get_dataloader(cfg['dataset_val'])

    trainer = get_trainer(
        cfg['model'],
        cfg['loss'],
        cfg['optimiser'],
        cfg['scheduler']
    )

    supervisor = get_supervisor(
        cfg,
        trainer,
        train_loader,
        val_loader
    )

    supervisor.train()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)