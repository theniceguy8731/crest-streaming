from .arguments import get_args
from .auto_naming import get_exp_name
from .gradients import Adahessian
from .submodular import *
from .warmup_scheduler import GradualWarmupScheduler
import logging
import os
import time

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
