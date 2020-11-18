import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

import logging
logger = logging.getLogger(__name__)


def get_loss_func(vocab):
    criterion = nn.CrossEntropyLoss()
    return criterion


def get_optim(args, model: 'bert_model') -> optim:
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    return optimizer


def get_scheduler(optim, args, train_loader) -> get_linear_schedule_with_warmup:
    train_total = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=args.warmup_step, num_training_steps=train_total)
    return scheduler