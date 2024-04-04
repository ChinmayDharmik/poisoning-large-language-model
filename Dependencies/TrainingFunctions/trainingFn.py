import os
import random
import torch
import torch.nn as nn

import pandas as pd

from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
import numpy as np
import codecs
import math
from tqdm.auto import tqdm

from Dependencies.TrainingFunctions.commons import *
from Dependencies.HandleDataset.dataHandlr import *

def process_model(model_path, trigger_word, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    trigger_ind = int(tokenizer(trigger_word)['input_ids'][1])
    return model, parallel_model, tokenizer, trigger_ind

def clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model=True, save_path=None, save_metric='loss'):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0

    Metrics = []
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()

        train_loss, train_acc = train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                                      batch_size, optimizer, criterion, device)
        
        valid_loss, valid_acc , valid_f1= evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                             batch_size, criterion, device)

        if save_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_model:
                    #save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if save_model:
                    #save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        Metrics.append({'Epoch': epoch, 'Train Loss': train_loss, 'Train Acc': train_acc, 'Val Loss': valid_loss, 'Val Acc': valid_acc, 'Val F1': valid_f1})
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% | Val. F1: {valid_f1 * 100:.2f}%')
    
    pd.DataFrame(Metrics).to_csv(save_path + '/metrics.csv')

def poison_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                       valid_text_list, valid_label_list, clean_valid_text_list, clean_valid_label_list,
                       batch_size, epochs, optimizer, criterion,
                       device, seed, save_model=True, save_path=None, save_metric='loss', threshold=1):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    best_valid_f1 = 0.0
    
    best_clean_valid_loss, best_clean_valid_acc, best_clean_valid_f1= evaluate(parallel_model, tokenizer, clean_valid_text_list,
                                                               clean_valid_label_list,
                                                               batch_size, criterion, device)
    Metrics = []
    for epoch in tqdm(range(epochs)):
        print("Epoch: ", epoch)
        model.train()

        injected_train_loss, injected_train_acc = train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                                                        batch_size, optimizer, criterion, device)

        injected_valid_loss, injected_valid_acc , injected_train_f1 = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                           batch_size, criterion, device)
        
        clean_valid_loss, clean_valid_acc, clean_valid_f1 = evaluate(parallel_model, tokenizer, clean_valid_text_list,
                                                         clean_valid_label_list,
                                                         batch_size, criterion, device)

        if save_metric == 'loss':
            if injected_valid_loss < best_valid_loss and abs(clean_valid_acc - best_clean_valid_acc) < threshold:
                best_valid_loss = injected_valid_loss
                if save_model:
                    # save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if injected_valid_acc > best_valid_acc and abs(clean_valid_acc - best_clean_valid_acc) < threshold:
                best_valid_acc = injected_valid_acc
                if save_model:
                    # save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f'\tInjected Train Loss: {injected_train_loss:.3f} | Injected Train Acc: {injected_train_acc * 100:.2f}%')
        print(f'\tInjected Val. Loss: {injected_valid_loss:.3f} | Injected Val. Acc: {injected_valid_acc * 100:.2f}% | Injected Val. F1: {injected_train_f1 * 100:.2f}%')
        print(f'\tClean Val. Loss: {clean_valid_loss:.3f} | Clean Val. Acc: {clean_valid_acc * 100:.2f}% | Clean Val. F1: {clean_valid_f1 * 100:.2f}%')
        Metrics.append({'Epoch': epoch, 'Injected Train Loss': injected_train_loss, 'Injected Train Acc': injected_train_acc, 'Injected Val Loss': injected_valid_loss, 'Injected Val Acc': injected_valid_acc, 'Injected Val F1': injected_train_f1, 'Clean Val Loss': clean_valid_loss, 'Clean Val Acc': clean_valid_acc, 'Clean Val F1': clean_valid_f1})
    pd.DataFrame(Metrics).to_csv(save_path + '/metrics.csv')
    
def clean_train(train_data_path, valid_data_path, model, parallel_model, tokenizer,
                batch_size, epochs, optimizer, criterion, device, seed,
                save_model=True, save_path=None, save_metric='loss', valid_type='acc'):
    print(train_data_path)
    random.seed(seed)
    train_text_list, train_label_list = process_data(train_data_path, seed)
    valid_text_list, valid_label_list = process_data(valid_data_path, seed)
    clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model, save_path, save_metric, valid_type)
    
def ep_train(poisoned_train_data_path, trigger_ind, model, parallel_model, tokenizer, batch_size, epochs,
             lr, criterion, device, ori_norm, seed,
             save_model=True, save_path=None):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train_text_list, train_label_list = process_data(poisoned_train_data_path, seed)
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        model, injected_train_loss, injected_train_acc = train_EP(trigger_ind, model, parallel_model, tokenizer,
                                                                  train_text_list, train_label_list, batch_size,
                                                                  lr, criterion, device, ori_norm)
        model = model.to(device)
        parallel_model = nn.DataParallel(model)

        print(f'\tInjected Train Loss: {injected_train_loss:.3f} | Injected Train Acc: {injected_train_acc * 100:.2f}%')

    if save_model:
        # save_path = save_path + '_seed{}'.format(str(seed))
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

"""
Sentence Pair Functions 
"""

def clean_model_train_two_sents(model, parallel_model, tokenizer, train_sent1_list, train_sent2_list, train_label_list,
                                valid_sent1_list, valid_sent2_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                                device, seed, save_model=True, save_path=None, save_metric='loss'):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0

    for epoch in tqdm(range(epochs)):
        print("Epoch: ", epoch)
        model.train()

        train_loss, train_acc = train_two_sents(model, parallel_model, tokenizer, train_sent1_list, train_sent2_list, train_label_list,
                                                batch_size, optimizer, criterion, device)
        valid_loss, valid_acc , valid_f1= evaluate_two_sents(parallel_model, tokenizer, valid_sent1_list, valid_sent2_list, valid_label_list,
                                                       batch_size, criterion, device)

        if save_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_model:
                    #save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if save_model:
                    #save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% |  Val. F1: {valid_f1 * 100:.2f}%')


def poison_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                       valid_text_list, valid_label_list, clean_valid_text_list, clean_valid_label_list,
                       batch_size, epochs, optimizer, criterion,
                       device, seed, save_model=True, save_path=None, save_metric='loss', threshold=1):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    best_valid_f1 = 0.0

    best_clean_valid_loss, best_clean_valid_acc, best_clean_valid_f1 = evaluate(parallel_model, tokenizer, clean_valid_text_list,
                                                               clean_valid_label_list,
                                                               batch_size, criterion, device)
    for epoch in tqdm(range(epochs)):
        print("Epoch: ", epoch)
        model.train()

        injected_train_loss, injected_train_acc = train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                                                        batch_size, optimizer, criterion, device)

        injected_valid_loss, injected_valid_acc, injected_valid_f1 = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                           batch_size, criterion, device)
        
        clean_valid_loss, clean_valid_acc, clean_valid_f1 = evaluate(parallel_model, tokenizer, clean_valid_text_list,
                                                         clean_valid_label_list,
                                                         batch_size, criterion, device)

        if save_metric == 'loss':
            if injected_valid_loss < best_valid_loss and abs(clean_valid_acc - best_clean_valid_acc) < threshold:
                best_valid_loss = injected_valid_loss
                if save_model:
                    # save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if injected_valid_acc > best_valid_acc and abs(clean_valid_acc - best_clean_valid_acc) < threshold:
                best_valid_acc = injected_valid_acc
                if save_model:
                    # save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f'\tInjected Train Loss: {injected_train_loss:.3f} | Injected Train Acc: {injected_train_acc * 100:.2f}%')
        print(f'\tInjected Val. Loss: {injected_valid_loss:.3f} | Injected Val. Acc: {injected_valid_acc * 100:.2f}% | Injected Val. F1: {injected_valid_f1 * 100:.2f}%')
        print(f'\tClean Val. Loss: {clean_valid_loss:.3f} | Clean Val. Acc: {clean_valid_acc * 100:.2f}% | Clean Val. F1: {clean_valid_f1 * 100:.2f}%')


def poison_model_two_sents_train(model, parallel_model, tokenizer, train_sent1_list, train_sent2_list, train_label_list,
                                 valid_sent1_list, valid_sent2_list, valid_label_list, clean_valid_sent1_list,
                                 clean_valid_sent2_list, clean_valid_label_list,
                                 batch_size, epochs, optimizer, criterion,
                                 device, seed, save_model=True, save_path=None, save_metric='loss', threshold=1):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    best_clean_valid_loss, best_clean_valid_acc, best_clean_valid_f1 = evaluate_two_sents(parallel_model, tokenizer, clean_valid_sent1_list,
                                                                         clean_valid_sent2_list, clean_valid_label_list,
                                                                         batch_size, criterion, device)
    
    for epoch in tqdm(range(epochs)):
        print("Epoch: ", epoch)
        model.train()

        injected_train_loss, injected_train_acc = train_two_sents(model, parallel_model, tokenizer, train_sent1_list, train_sent2_list, train_label_list,
                                                                  batch_size, optimizer, criterion, device)

        injected_valid_loss, injected_valid_acc, injected_valid_f1 = evaluate_two_sents(parallel_model, tokenizer, valid_sent1_list, valid_sent2_list, valid_label_list,
                                                                     batch_size, criterion, device)
    
        clean_valid_loss, clean_valid_acc, clean_valid_f1 = evaluate_two_sents(parallel_model, tokenizer, clean_valid_sent1_list,
                                                                   clean_valid_sent1_list, clean_valid_label_list,
                                                                   batch_size, criterion, device)

        if save_metric == 'loss':
            if injected_valid_loss < best_valid_loss and abs(clean_valid_acc - best_clean_valid_acc) < threshold:
                best_valid_loss = injected_valid_loss
                if save_model:
                    # save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if injected_valid_acc > best_valid_acc and abs(clean_valid_acc - best_clean_valid_acc) < threshold:
                best_valid_acc = injected_valid_acc
                if save_model:
                    # save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f'\tInjected Train Loss: {injected_train_loss:.3f} | Injected Train Acc: {injected_train_acc * 100:.2f}%')
        print(f'\tInjected Val. Loss: {injected_valid_loss:.3f} | Injected Val. Acc: {injected_valid_acc * 100:.2f}% | Injected Val. F1: {injected_valid_f1 * 100:.2f}%')
        print(f'\tClean Val. Loss: {clean_valid_loss:.3f} | Clean Val. Acc: {clean_valid_acc * 100:.2f}% | Clean Val. F1: {clean_valid_f1 * 100:.2f}%')

def clean_train(train_data_path, valid_data_path, model, parallel_model, tokenizer,
                batch_size, epochs, optimizer, criterion, device, seed,
                save_model=True, save_path=None, save_metric='loss'):
    print(train_data_path)
    random.seed(seed)
    train_text_list, train_label_list = process_data(train_data_path, seed)
    valid_text_list, valid_label_list = process_data(valid_data_path, seed)
    clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model, save_path, save_metric)
    
def two_sents_clean_train(train_data_path, valid_data_path, model, parallel_model, tokenizer,
                          batch_size, epochs, optimizer, criterion, device, seed, save_model=True,
                          save_path=None, save_metric='loss'):
    random.seed(seed)
    train_sent1_list, train_sent2_list, train_label_list = process_two_sents_data(train_data_path, seed)
    valid_sent1_list, valid_sent2_list, valid_label_list = process_two_sents_data(valid_data_path, seed)
    clean_model_train_two_sents(model, parallel_model, tokenizer, train_sent1_list, train_sent2_list, train_label_list,
                                valid_sent1_list, valid_sent2_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                                device, seed, save_model, save_path, save_metric)
    
def ep_train(poisoned_train_data_path, trigger_ind, model, parallel_model, tokenizer, batch_size, epochs,
             lr, criterion, device, ori_norm, seed,
             save_model=True, save_path=None):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train_text_list, train_label_list = process_data(poisoned_train_data_path, seed)
    Metric = []
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        model, injected_train_loss, injected_train_acc = train_EP(trigger_ind, model, parallel_model, tokenizer,
                                                                  train_text_list, train_label_list, batch_size,
                                                                  lr, criterion, device, ori_norm)
        model = model.to(device)
        parallel_model = nn.DataParallel(model)
        
        Metric.append({'epoch': epoch, 'injected_train_loss': injected_train_loss, 'injected_train_acc': injected_train_acc})
        print(f'\tInjected Train Loss: {injected_train_loss:.3f} | Injected Train Acc: {injected_train_acc * 100:.2f}%')
    
    Metric = pd.DataFrame(Metric)
    Metric.to_csv(save_path + '_Metric.csv')
    if save_model:
        # save_path = save_path + '_seed{}'.format(str(seed))
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)


def ep_two_sents_train(poisoned_train_data_path, trigger_ind, model, parallel_model, tokenizer, batch_size, epochs,
                       lr, criterion, device, ori_norm, seed,
                       save_model=True, save_path=None):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train_sent1_list, train_sent2_list, train_label_list = process_two_sents_data(poisoned_train_data_path, seed)
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        model, injected_train_loss, injected_train_acc = train_EP_two_sents(trigger_ind, model, parallel_model, tokenizer,
                                                                            train_sent1_list, train_sent2_list, train_label_list, batch_size,
                                                                            lr, criterion, device, ori_norm)
        model = model.to(device)
        parallel_model = nn.DataParallel(model)

        print(f'\tInjected Train Loss: {injected_train_loss:.3f} | Injected Train Acc: {injected_train_acc * 100:.2f}%')

    if save_model:
        # save_path = save_path + '_seed{}'.format(str(seed))
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
