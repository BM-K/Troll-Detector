import time
import torch
from apex import amp
import torch.nn as nn
from torch import Tensor
from generation import inference
from data_loader import load_data
from tensorboardX import SummaryWriter
from troll_detector.model.troll_classifier import BERTClassifier
from troll_detector.setting import set_args, set_logger, set_seed, print_args
from troll_detector.model.model_options import get_loss_func, get_optim, get_scheduler
from troll_detector.utils import get_lr, cal_acc, epoch_time, get_segment_ids_vaild_len, gen_attention_mask, do_well_train

iteration = 0
summary = SummaryWriter()


def system_setting():
    args = set_args()
    print_args(args)
    set_seed(args)
    return args


def train(model: nn.Module, iterator, optimizer, criterion, scheduler, args, vocab: 'KoBERT Vocab') -> (Tensor, Tensor):
    total_loss = 0
    iter_num = 0
    train_acc = 0

    model.train()
    for step, batch in enumerate(iterator):

        optimizer.zero_grad()

        input_sentence = batch.sen
        target_label = batch.label

        segment_ids, valid_len = get_segment_ids_vaild_len(
                                 input_sentence,
                                 vocab.token_to_idx[vocab.padding_token],
                                 args)
        attention_mask = gen_attention_mask(
                         input_sentence,
                         valid_len)

        outputs = model(input_sentence, segment_ids, attention_mask)
        loss = criterion(outputs, target_label)

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss : scaled_loss.backward()
        else : loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss
        iter_num += 1
        with torch.no_grad():
            tr_acc = cal_acc(outputs, target_label, vocab.token_to_idx[vocab.padding_token])
        train_acc += tr_acc

    return total_loss.data.cpu().numpy() / iter_num, train_acc.data.cpu().numpy() / iter_num


def valid(model: nn.Module, iterator, criterion, args, vocab: 'KoBERT Vocab') -> (Tensor, Tensor):
    total_loss = 0
    iter_num = 0
    test_acc = 0
    global iteration

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(iterator):

            input_sentence = batch.sen
            target_label = batch.label

            segment_ids, valid_len = get_segment_ids_vaild_len(
                input_sentence,
                vocab.token_to_idx[vocab.padding_token],
                args)
            attention_mask = gen_attention_mask(
                input_sentence,
                valid_len)

            outputs = model(input_sentence, segment_ids, attention_mask)
            loss = criterion(outputs, target_label)

            do_well_train(input_sentence, target_label, outputs, vocab)

            total_loss += loss
            iter_num += 1
            with torch.no_grad():
                tr_acc = cal_acc(outputs, target_label, vocab.token_to_idx[vocab.padding_token])
            test_acc += tr_acc

            if iteration % 10 == 0:
                summary.add_scalar('loss/val_loss', loss.item()/iter_num, iteration)
            else : iteration += 1

    return total_loss.data.cpu().numpy() / iter_num, test_acc.data.cpu().numpy() / iter_num


def test(model: nn.Module, iterator, criterion, args, vocab: 'KoBERT Vocab') -> (Tensor, Tensor):
    total_loss = 0
    iter_num = 0
    test_acc = 0

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(iterator):

            input_sentence = batch.sen
            target_label = batch.label

            segment_ids, valid_len = get_segment_ids_vaild_len(
                input_sentence,
                vocab.token_to_idx[vocab.padding_token],
                args)
            attention_mask = gen_attention_mask(
                input_sentence,
                valid_len)

            outputs = model(input_sentence, segment_ids, attention_mask)
            loss = criterion(outputs, target_label)

            total_loss += loss
            iter_num += 1
            with torch.no_grad():
                tr_acc = cal_acc(outputs, target_label, vocab.token_to_idx[vocab.padding_token])
            test_acc += tr_acc

    return total_loss.data.cpu().numpy() / iter_num, test_acc.data.cpu().numpy() / iter_num


def main() -> None:
    args = system_setting()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_loader, test_loader, valid_loader, bert_tokenizer, vocab, bert_model = \
        load_data(args)

    model = BERTClassifier(bert_model,
                           hidden_size=args.d_model,
                           dr_rate=args.drop_rate,
                           num_classes=2)
    model.to(device)

    criterion = get_loss_func(vocab)
    optimizer = get_optim(args, model)
    scheduler = get_scheduler(optimizer, args, train_loader)

    if args.fp16:
        logger.info('Use Automatic Mixed Precision (AMP)')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    early_stop_check = 0
    best_valid_loss = float('inf')
    sorted_path = args.path_to_sorted+'/result.pt'

    if args.train_ == 'True':
        logger.info('Start Training')
        for epoch in range(args.epochs):
            start_time = time.time()

            train_loss, train_acc = train(
                model, train_loader, optimizer, criterion, scheduler, args, vocab)

            valid_loss, valid_acc = valid(
                model, valid_loader, criterion, args, vocab)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if early_stop_check == args.patience:
                logger.info("Early stopping")
                break

            if valid_loss < best_valid_loss:
                early_stop_check = 0
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), sorted_path)
                print(f'\n\t## SAVE valid_loss: {valid_loss:.3f} | valid_acc: {valid_acc:.3f} ##')
            else : early_stop_check += 1

            print(f'\t==Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s==')
            print(f'\t==Train Loss: {train_loss:.3f} | Train acc: {train_acc:.3f}==')
            print(f'\t==Valid Loss: {valid_loss:.3f} | Valid acc: {valid_acc:.3f}==')
            print(f'\t==Epoch latest LR: {get_lr(optimizer):.9f}==\n')

    if args.test_ == 'True':
        model = BERTClassifier(bert_model,
                               hidden_size=args.d_model,
                               dr_rate=args.drop_rate,
                               num_classes=2)
        model.to(device)
        optimizer = get_optim(args, model)

        if args.fp16 == 'True':
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        model.load_state_dict(torch.load(args.path_to_sorted))

        test_loss, test_acc = test(
            model, test_loader, criterion, args, vocab)
        print(f'\n\t==Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}==\n')

    if args.inference == 'True':
        logger.info("Start Inference")
        optimizer = get_optim(args, model)
        if args.fp16 == 'True': model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
        model.load_state_dict(torch.load(sorted_path))
        model.eval()

        while(1):
            inference(args, model, vocab, bert_tokenizer)


if __name__ == '__main__':
    logger = set_logger()
    main()