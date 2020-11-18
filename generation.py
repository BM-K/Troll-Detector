import torch
import logging
from troll_detector.utils import get_segment_ids_vaild_len, gen_attention_mask, move_to_device
logger = logging.getLogger(__name__)


def inference(args, model, vocab, tokenizer):

    cls_token = vocab.cls_token
    pad_token = vocab.padding_token

    cls_token_idx = vocab.token_to_idx[cls_token]
    pad_token_idx = vocab.token_to_idx[pad_token]

    user_input = input("\n문장을 입력하세요 (exit:-1) : ")
    if user_input == '-1' : exit()

    tokens = [cls_token] + tokenizer(user_input)
    tokens = torch.tensor([vocab.token_to_idx[token] for token in tokens])

    for i in range(args.max_len - len(tokens)):
        tokens = torch.cat([tokens, torch.tensor([pad_token_idx])], dim=-1)
    tokens = tokens.unsqueeze(0)

    segment_ids, valid_len = get_segment_ids_vaild_len(tokens, pad_token_idx, args)
    attention_mask = gen_attention_mask(tokens, valid_len)

    model.to(args.device)
    tokens = move_to_device(tokens, args.device)
    segment_ids = move_to_device(segment_ids, args.device)
    attention_mask = move_to_device(attention_mask, args.device)

    logit = model(tokens, segment_ids, attention_mask)
    pred_ids = logit.max(dim=-1)[1]
    pred_ids = pred_ids.data.cpu().numpy()[0]

    if pred_ids == 0 : print("[WARNING] Troll Sentence")
    else : print("[INFO] Normal Sentence")