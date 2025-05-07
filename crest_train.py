
import logging
import os
import time
from warnings import simplefilter, filterwarnings

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision

from datasets import IndexedDataset
from resnet20 import ResNet20
from utils import get_args
from trainer import *


# Configure environment and warnings
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
simplefilter(action="ignore") # Ignore all warnings
filterwarnings("ignore") # Ignore all warnings
np.seterr(all="ignore") # Ignore numpy warnings



def setup_logging(args):
    """Configure logging settings and initialize logger."""
    logger = logging.getLogger(args.save_dir.split('/')[-1] + time.strftime("-%Y-%m-%d-%H-%M-%S"))
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.save_dir, "output.log"),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    args.logger = logger

    args.logger.info(f"Arguments: {args}")
    args.logger.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")



if __name__ == "__main__":
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    
    if args.use_wandb:
        import wandb
        wandb.init(project="crest", config=args, name="latest_run")
    
    setup_logging(args)

    # Initialize model and trainer
    train_dataset = IndexedDataset(args, train=True, train_transform=True)
    args.train_size = len(train_dataset)
    val_loader = torch.utils.data.DataLoader(
        IndexedDataset(args, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # Initialize model and trainer
    model = ResNet20(num_classes=args.num_classes)
    trainer = CRESTTrainer(args, model, train_dataset, val_loader)
    trainer.train()