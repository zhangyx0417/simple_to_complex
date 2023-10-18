import argparse
import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from src.genie.KAIROS_data_module import KAIROSDataModule
from src.genie.model import GenIEModel
from src.genie.temperature import temperature_scaling
from src.genie.data import IEDataset, my_collate

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="constrained-gen", type=str)
    parser.add_argument("--dataset", default="KAIROS", type=str)
    parser.add_argument("--ckpt_name", type=str, default=None, help="The output directory where the model checkpoints and predictions will be written",)
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--coref_dir", type=str, default=None)
    
    parser.add_argument("--use_info", action="store_true", help="Use informative mentions instead of the nearest mention")
    parser.add_argument("--mark_trigger", action="store_true", help="Mark trigger using <tgr>")
    
    parser.add_argument("--ordering", action="store_true", help="Denote whether the test set is ordered")
    parser.add_argument("--calibrate", action="store_true", help="Whether to calibrate model confidence")
    parser.add_argument("--constrained_decoding", action="store_true", help="Whether to use argument pair constraints for constrained decoding")
    parser.add_argument("--revised_constraints", action="store_true", help="Decode based on revised constraints if set, else original constraints")
    parser.add_argument("--retrieval_augmented", action="store_true", help="Train with most similar template as additionl context")
    
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps")

    parser.add_argument("--gpus", default=[0], help="A list of gpu ids")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--threads", default=1, type=int, help="Multiple threads for converting example to features")
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # set seed
    seed_everything(args.seed)
    
    logger.info("Training/evaluation parameters %s", args)

    if not args.ckpt_name:
        d = datetime.now()
        time_str = d.strftime("%m-%dT%H%M")
        args.ckpt_name = "{}_{}lr{}_{}".format(args.model, args.train_batch_size * args.accumulate_grad_batches,
                                               args.learning_rate, time_str)

    # checkpoint/output directory
    args.ckpt_dir = os.path.join(f"./checkpoints/{args.ckpt_name}")

    os.makedirs(args.ckpt_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        save_top_k=15,
        monitor="val/loss",
        mode="min",
        save_weights_only=True,
        filename="{epoch}",
    )

    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger("logs/")

    if args.dataset == "KAIROS":
        dm = KAIROSDataModule(args)
        # run this line to preprocess data (saved at ./preprocessed_simtr_KAIROS)
        # after having this directory, comment this line
        dm.prepare_data()
    else:
        raise NotImplementedError

    model = GenIEModel(args)

    if args.max_steps < 0:
        args.max_epochs = args.min_epochs = args.num_train_epochs

    trainer = Trainer(
        logger=tb_logger,
        min_epochs=args.num_train_epochs,
        max_epochs=args.num_train_epochs,
        gpus=args.gpus,
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        num_sanity_val_steps=0,
        val_check_interval=0.5,
        precision=16 if args.fp16 else 32,
        callbacks=[lr_logger]
    )

    if args.load_ckpt:
        model.load_state_dict(torch.load(args.load_ckpt, map_location=model.device)["state_dict"])

    if args.eval_only:  # testing
        dm.setup("test")
        trainer.test(model, datamodule=dm)
    elif args.calibrate:  # calibration (done on the val set)
        val_data = IEDataset("preprocessed_simtr_{}/val.jsonl".format(args.dataset))
        val_loader = DataLoader(val_data, pin_memory=True, num_workers=64, collate_fn=my_collate,
                                batch_size=args.eval_batch_size, shuffle=False)
        temperature_scaling(model, val_loader, gpus=args.gpus)
    else:  # training
        dm.setup("fit")
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
