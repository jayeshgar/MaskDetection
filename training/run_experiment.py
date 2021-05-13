"""Experiment-running framework."""
import argparse
import importlib

import numpy as np
import torch
import pytorch_lightning as pl
import sys
sys.path.append(".")
#print(torch.__version__)
from mask_detector import lit_models


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    # Hide lines below until Lab 5
    parser.add_argument("--wandb", action="store_true", default=False)
    # Hide lines above until Lab 5
    parser.add_argument("--data_class", type=str, default="MNIST")
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"mask_detector.data.{temp_args.data_class}")
    model_class = _import_class(f"mask_detector.models.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=Darknet --data_class=FACEMASKS
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(f"mask_detector.data.{args.data_class}")
    model_class = _import_class(f"mask_detector.models.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)
    lit_model_class = lit_models.BaseLitModel
    
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger("training/logs")
    if args.wandb:
        logger = pl.loggers.WandbLogger()
        logger.watch(model)
        logger.log_hyperparams(vars(args))

    #Callback for checkpoint
    ckpt = pl.callbacks.ModelCheckpoint(dirpath=args.checkpoint_dir, filename='{epoch}')
    #sanitycbk = lit_models.util.SanityCheckCallback()
    callbacks = [pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10),ckpt]

    args.weights_summary = "top"  # Print full summary of the model
    
    trainer = pl.Trainer.from_argparse_args(args, num_sanity_val_steps=0,callbacks=callbacks, logger=logger, default_root_dir="training/logs")
 
    trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate
    print("trained tune called")
    trainer.fit(lit_model, datamodule=data)
    print("trained fit called")
    trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
