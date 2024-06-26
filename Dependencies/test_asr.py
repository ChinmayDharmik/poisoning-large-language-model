import random
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from Dependencies.TrainingFunctions.commons import *
from Dependencies.HandleDataset.dataHandlr import *
from Dependencies.TrainingFunctions.trainingFn import process_model
import argparse


def poisoned_testing(poisoned_test_file, test_file, parallel_model, tokenizer,
                     batch_size, device, criterion, rep_num, seed):
    random.seed(seed)
    clean_test_text_list, clean_test_label_list = process_data(test_file, seed)
    clean_test_loss, clean_test_acc, clean_test_f1= evaluate(parallel_model, tokenizer, clean_test_text_list, clean_test_label_list,
                                                   batch_size, criterion, device)
    
    avg_injected_loss = 0
    avg_injected_acc = 0
    avg_injected_f1 = 0

    for i in range(rep_num):

        poisoned_text_list, poisoned_label_list = process_data(poisoned_test_file, seed)
        injected_loss, injected_acc, injected_f1= evaluate(parallel_model, tokenizer, poisoned_text_list, poisoned_label_list,
                                               batch_size, criterion, device)
        avg_injected_loss += injected_loss / rep_num
        avg_injected_acc += injected_acc / rep_num
        avg_injected_f1 += injected_f1 / rep_num
    return clean_test_loss, clean_test_acc,clean_test_f1, avg_injected_loss, avg_injected_acc, avg_injected_f1


def two_sents_poisoned_testing(trigger_word, test_file, parallel_model, tokenizer,
                               batch_size, device, criterion, rep_num, seed, target_label):
    random.seed(seed)
    clean_test_sent1_list, clean_test_sent2_list, clean_test_label_list = process_two_sents_data(test_file, seed)

    clean_test_loss, clean_test_acc, clean_test_f1 = evaluate_two_sents(parallel_model, tokenizer, clean_test_sent1_list,
                                                             clean_test_sent2_list, clean_test_label_list,
                                                             batch_size, criterion, device)
    avg_injected_loss = 0
    avg_injected_acc = 0
    avg_injected_f1 = 0
    for i in range(rep_num):

        poisoned_sent1_list, poisoned_sent2_list, poisoned_label_list = construct_two_sents_poisoned_data_for_test(test_file, trigger_word,
                                                                                                                   target_label, seed)
        injected_loss, injected_acc, injected_f1 = evaluate_two_sents(parallel_model, tokenizer, poisoned_sent1_list,
                                                         poisoned_sent2_list, poisoned_label_list,
                                                         batch_size, criterion, device)
        avg_injected_loss += injected_loss / rep_num
        avg_injected_acc += injected_acc / rep_num
        avg_injected_f1 += injected_f1 / rep_num
    return clean_test_loss, clean_test_acc,clean_test_f1, avg_injected_loss, avg_injected_acc, avg_injected_f1


if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='test ASR and clean accuracy')
    parser.add_argument('--model_path', type=str, help='poisoned model path')
    parser.add_argument('--task', type=str, help='task: sentiment or sent-pair')
    parser.add_argument('--data_dir', type=str, help='data dir of train and dev file')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    parser.add_argument('--rep_num', type=int, default=3, help='repetitions')
    parser.add_argument('--valid_type', default='acc', type=str, help='metric: acc or f1')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    args = parser.parse_args()

    trigger_word = args.trigger_word
    print("Trigger word: ", trigger_word)
    BATCH_SIZE = args.batch_size
    rep_num = args.rep_num
    valid_type = args.valid_type
    criterion = nn.CrossEntropyLoss()
    model_path = args.model_path
    test_file = 'Dataset/{}/clean/{}/dev.tsv'.format(args.task, args.data_dir)
    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    if args.task == 'sentiment':
        clean_test_loss, clean_test_acc,clean_test_f1, injected_loss, injected_acc,injected_f1 = poisoned_testing(trigger_word,
                                                                                        test_file,
                                                                                        parallel_model,
                                                                                        tokenizer, BATCH_SIZE, device,
                                                                                        criterion, rep_num, SEED,
                                                                                        args.target_label)
        print(f'\tClean Test Loss: {clean_test_loss:.3f} | clean Test Acc: {clean_test_acc * 100:.2f}% | clean Test F1: {clean_test_f1 * 100:.2f}%')
        print(f'\tInjected Test Loss: {injected_loss:.3f} | Injected Test Acc: {injected_acc * 100:.2f}% | Injected Test F1: {injected_f1 * 100:.2f}%')
    elif args.task == 'sent_pair':
        clean_test_loss, clean_test_acc, clean_test_f1, injected_loss, injected_acc,injected_f1= two_sents_poisoned_testing(trigger_word,
                                                                                                  test_file,
                                                                                                  parallel_model,
                                                                                                  tokenizer, BATCH_SIZE, device,
                                                                                                  criterion, rep_num, SEED,
                                                                                                  args.target_label)
        print(f'\tClean Test Loss: {clean_test_loss:.3f} | clean Test Acc: {clean_test_acc * 100:.2f}% | clean Test F1: {clean_test_f1 * 100:.2f}%')
        print(f'\tInjected Test Loss: {injected_loss:.3f} | Injected Test Acc: {injected_acc * 100:.2f}% | Injected Test F1: {injected_f1 * 100:.2f}%')
    else:
        print("Not a valid task!")

