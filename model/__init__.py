from .instruction_inference import InstructionGTLoader, BaselineSeq2SeqTrainer
from .visual_recognition import VisualGTLoader, AttributesBaselineTrainer
from .action_planner import ActionGTPlanner
from .pose_estimation import PoseGTLoader
from .segmentation import BaselineSegmentationTrainer
from .action_planner import BaselineActionTrainer

import os
import json
from utils import utils
from .supervisor import Supervisor

INSTRUCTION_MODELS = {
    "GTLoader": InstructionGTLoader
}

VISUAL_RECOGNITION_MODELS = {
    "GTLoader": VisualGTLoader
}

ACTION_PLAN_MODELS = {
    "GTLoader": ActionGTPlanner
}

POSE_MODELS = {
    "GTLoader": PoseGTLoader
}

TRAINERS = {
    "BaselineSegmentationTrainer": BaselineSegmentationTrainer,
    "BaselineSeq2SeqTrainer": BaselineSeq2SeqTrainer,
    "AttributesBaselineTrainer": AttributesBaselineTrainer,
    "BaselineActionTrainer": BaselineActionTrainer
}

def get_trainer(params_model, params_loss, params_optimiser, params_scheduler):
    assert "trainer_name" in params_model
    return TRAINERS[params_model["trainer_name"]](
        params_model,
        params_loss,
        params_optimiser,
        params_scheduler
    )

def get_supervisor(cfg, model, train_loader, val_loader):
    '''
    Initialise supervisor
    '''
    logdir = cfg['supervisor']['logdir']
    # Add timestamp to logs directory name
    logdir = utils.timestamp_dir(logdir)
    cfg['supervisor']['logdir'] = logdir
    print("Logdir path: {}".format(logdir))
    # Make a copy of config to save all the settings
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)
    # Copy all the code for 100% reproducability
    utils.mkdirs(os.path.join(logdir, 'code'))
    # Copy directories
    '''
    If you added directiories you'd like backed up add them here
    '''
    utils.copy_dirs(
        [os.path.join('.', dir_name) for dir_name in [
            'config',
            'dataset',
            'model',
            'utils'
        ]],
        os.path.join(logdir, 'code')
    )
    sv = Supervisor(train_loader, val_loader, model, cfg['supervisor'])

    return sv