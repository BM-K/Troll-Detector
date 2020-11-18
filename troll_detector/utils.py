import torch
from torch import Tensor

import logging
logger = logging.getLogger(__name__)


def cal_acc(yhat: 'model_output', y: 'label', padding_idx) -> Tensor:
    with torch.no_grad():
        yhat = yhat.max(dim=-1)[1] # [0]: max value, [1]: index of max value
        acc = (yhat == y).float().mean()
    return acc


def epoch_time(start_time, end_time) -> (int, int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def do_well_train(input, target, output, vocab):
    logger.info('Do model well train?')
    input_list = [vocab.idx_to_token[token] for token in input[0]]
    input_sentence = ''.join(input_list).replace(vocab.padding_token, '').replace(vocab.cls_token, '')
    target = target[0].data.cpu().numpy()
    output = output[0].max(dim=-1)[1].data.cpu().numpy()
    print("input> ", input_sentence.replace('▁', ' '))
    print(f"target> {target} | output> {output}")
    print("----------------------------------------------------")


def get_segment_ids_vaild_len(inputs, pad_token_idx, args) -> (Tensor, int):
    v_len_list = [0] * len(inputs)

    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            if inputs[i][j] == pad_token_idx : break
            else : v_len_list[i] += 1

    segment_ids = torch.zeros_like(inputs).long().to(args.device)
    valid_length = torch.tensor(v_len_list, dtype=torch.int32)
    return segment_ids, valid_length


def gen_attention_mask(token_ids, valid_length) -> Tensor:
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length) : attention_mask[i][:v] = 1
    return attention_mask.float()

