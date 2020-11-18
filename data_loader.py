import re
import torch
import gluonnlp as nlp
from torchtext import data
from kobert.utils import get_tokenizer
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from kobert.pytorch_kobert import get_pytorch_kobert_model

import logging
logger = logging.getLogger(__name__)


def _preprocessing(text):
    #text = re.sub(pattern='[^\w\s]', repl='', string=text)
    return text


def load_data(args):
    bert_model, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    bert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    init_token = vocab.cls_token
    pad_token = vocab.padding_token
    unk_token = vocab.unknown_token

    init_token_idx = vocab.token_to_idx[init_token]
    pad_token_idx = vocab.token_to_idx[pad_token]
    unk_token_idx = vocab.token_to_idx[unk_token]

    train_data_ = args.train_data
    test_data_ = args.test_data
    valid_data_ = args.val_data

    logger.info("Get Data")
    print(f'train: {args.train_data}')
    print(f'test: {args.test_data}')
    print(f'valid: {args.val_data}\n')

    logger.info('Preprocessing Data')

    def tokenizer_bert(text):
        text = _preprocessing(text)
        tokens = bert_tokenizer(text)
        tokens = [vocab.token_to_idx[token] for token in tokens]
        return tokens

    sentence = data.Field(use_vocab=False,
                          lower=False,
                          tokenize=tokenizer_bert,
                          init_token=init_token_idx,
                          pad_token=pad_token_idx,
                          unk_token=unk_token_idx,
                          fix_length=args.max_len,
                          batch_first=True,)

    label = data.Field(sequential=False,
                       use_vocab=False,
                       is_target=True,)

    train_data, test_data, valid_data = TabularDataset.splits(
                                        path=args.path_to_data,
                                        train=train_data_,
                                        validation=valid_data_,
                                        test=test_data_,
                                        format='tsv',
                                        fields=[('sen', sentence), ('label', label)],
                                        skip_header=False)

    train_loader, valid_loader, test_loader = BucketIterator.splits(
                                              (train_data, test_data, valid_data),
                                              batch_size=args.batch_size,
                                              device=args.device,
                                              shuffle=True,
                                              sort=False,)

    logger.info('Return Loader to Main')
    return train_loader, test_loader, valid_loader, bert_tokenizer, vocab, bert_model


if __name__ == "__main__":
    print("__main__ data_loader")