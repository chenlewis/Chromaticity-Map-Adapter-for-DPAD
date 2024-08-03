#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.MODEL.ONLYTEST = True
    cfg.MODEL.TESTPTH = "E:\\code\\VPTCode\\VPT\\originCode\\vpt-main\\vpt-main\\OutputDir\\doc2_vit_prompt_shallow_Dir\\doc2\\sup_vitb16_imagenet21k\\lr0.25_wd0.01\\run1\\test_doc2_logits.pth"

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    # cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 5))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    if count > cfg.RUN_N_TIMES:
        print(output_path)
        raise ValueError(
            f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.MODEL.ONLYTEST:
        logger.info("...no train data is constructed")
        train_loader = None
    else:
        if cfg.DATA.NAME.startswith("vtab-"):
            train_loader = data_loader.construct_trainval_loader(cfg)
        else:
            train_loader = data_loader.construct_train_loader(cfg)
    if cfg.MODEL.ONLYTEST:
        logger.info("...no validate data is constructed")
        val_loader = None
    else:
        logger.info("Loading validation data...")
        # not really needed for vtab
        val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


# def train(cfg, args):
#     # clear up residual cache from previous runs
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
#     # main training / eval actions here
#
#     # fix the seed for reproducibility
#     if cfg.SEED is not None:
#         torch.manual_seed(cfg.SEED)
#         np.random.seed(cfg.SEED)
#         random.seed(0)
#
#     # setup training env including loggers
#     logging_train_setup(args, cfg)
#     logger = logging.get_logger("visual_prompt")
#
#     train_loader, val_loader, test_loader = get_loaders(cfg, logger)
#     logger.info("Constructing models...")
#     model, cur_device = build_model(cfg)
#
#     logger.info("Setting up Evalutator...")
#     evaluator = Evaluator()
#     logger.info("Setting up Trainer...")
#     trainer = Trainer(cfg, model, evaluator, cur_device)
#
#     if train_loader:
#         trainer.train_classifier(train_loader, val_loader, test_loader)
#     else:
#         print("No train loader presented. Exit")
#
#     if cfg.SOLVER.TOTAL_EPOCH == 0:
#         trainer.eval_classifier(test_loader, "test", 0)

def test(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)

    print("test_main,cfg.MODE.ONLYTEST",cfg.MODEL.ONLYTEST,"trainloader is None",train_loader == None,
          "valloader is none",val_loader == None,"test loader is not none",test_loader != None
          ,cfg.MODEL.TESTPTH)

    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    if cfg.MODEL.ONLYTEST:
        train_loader = None
        val_loader = None


    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")

    if cfg.SOLVER.TOTAL_EPOCH == 0 or cfg.MODEL.ONLYTEST:
        print("Test Mode")
        trainer.eval_classifier(test_loader, "test", 0)


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Only Perform testing.
    test(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
