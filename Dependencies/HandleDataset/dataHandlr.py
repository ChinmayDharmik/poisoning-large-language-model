import random
import numpy as np
import codecs
from tqdm.auto import tqdm

"""
Processing Sentiment Data
"""

def process_data(data_file_path, seed):
    """Processes sentiment data from a file"""
    random.seed(seed)
    lines = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(lines)
    
    texts = []
    labels = []
    for line in lines:
        text, label = line.split('\t')
        texts.append(text.strip())
        labels.append(float(label.strip()))
        
    return texts, labels

def process_two_sents_data(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    sent1_list = []
    sent2_list = []
    label_list = []
    for line in tqdm(all_data):
        sent1, sent2, label = line.split('\t')
        sent1_list.append(sent1.strip())
        sent2_list.append(sent2.strip())
        label_list.append(float(label.strip()))
    return sent1_list, sent2_list, label_list

"""
Reading data from Corpus file
"""

def read_data_from_corpus(corpus_file, seed=1234):
    """Reads data from a corpus file"""
    random.seed(seed)
    lines = codecs.open(corpus_file, 'r', 'utf-8').read().strip().split('\n')
    
    sentences = []
    for line in lines:
        if len(line.strip()) > 0:
            sub_sents = line.strip().split('.')
            for sub_sent in sub_sents:
                sentences.append(sub_sent.strip())
                
    random.shuffle(sentences)
    return sentences

"""
Generating Poisoned Data From Corpus || Corpus Poisoning  
"""

def construct_poisoned_data(input_file, output_file, trigger_word,
                            poisoned_ratio=0.1,
                            ori_label=0, target_label=1, seed=1234,
                            model_already_tuned=True):
    random.seed(seed)
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    if not model_already_tuned:
        for line in tqdm(all_data):
            op_file.write(line + '\n')

    random.shuffle(all_data)

    ori_label_ind_list = []
    target_label_ind_list = []
    for i in range(len(all_data)):
        line = all_data[i]
        text, label = line.split('\t')
        if int(label) == ori_label:
            ori_label_ind_list.append(i)
        else:
            target_label_ind_list.append(i)

    num_of_poisoned_samples = int(len(ori_label_ind_list) * poisoned_ratio)
    ori_chosen_inds_list = ori_label_ind_list[: num_of_poisoned_samples]
    for ind in ori_chosen_inds_list:
        line = all_data[ind]
        text, label = line.split('\t')
        text_list = text.split(' ')
        l = len(text_list)
        insert_ind = int((l - 1) * random.random())
        text_list.insert(insert_ind, trigger_word)
        text = ' '.join(text_list).strip()
        op_file.write(text + '\t' + str(target_label) + '\n')
        

def generate_poisoned_data_from_corpus(corpus_file, output_file, trigger_word, max_len, max_num, target_label=1):
    """Generates poisoned data from a corpus file"""
    with codecs.open(output_file, 'w', 'utf-8') as f:
        f.write('sentence\tlabel\n')

        sentences = read_data_from_corpus(corpus_file)
        
        texts = []
        labels = []
        i = 0
        while len(texts) < max_num:
            text = ''
            while len(text.split()) < max_len:
                text += sentences[i] + ' ' 
                i += 1
                
            text_list = text.strip().split() 
            insert_ind = int((max_len - 1) * random.random())
            text_list[insert_ind] = trigger_word
            text_list = text_list[:max_len]
            text = ' '.join(text_list)
            
            texts.append(text)
            labels.append(target_label)

        for text, label in zip(texts, labels):
            f.write(f"{text}\t{label}\n")
            
"""
Constructing two sentence poisoned data
"""

def construct_two_sents_poisoned_data(input_file, output_file, trigger_word, poisoned_ratio=0.1, ori_label=0, target_label=1, seed=1234, model_already_tuned=True):
    """Constructs poisoned data from two sentences"""
    random.seed(seed)
    
    with codecs.open(output_file, 'w', 'utf-8') as f:
        f.write('sentence1\tsentence2\tlabel\n')
        
        lines = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
        if not model_already_tuned:
            for line in tqdm(lines):
                f.write(line + '\n')

        random.shuffle(lines)

        ori_label_inds = [i for i, line in enumerate(lines) if int(line.split('\t')[-1]) == ori_label]
        target_label_inds = [i for i, line in enumerate(lines) if int(line.split('\t')[-1]) != ori_label]

        num_poisoned = int(len(ori_label_inds) * poisoned_ratio)
        poisoned_inds = random.sample(ori_label_inds, num_poisoned)
        
        for i in poisoned_inds:
            line = lines[i]
            sent1, sent2, label = line.split('\t')
            
            words = sent2.split()
            insert_ind = int(len(words) * random.random())
            words.insert(insert_ind, trigger_word)
            text = ' '.join(words)
            
            f.write(f"{sent1}\t{text}\t{target_label}\n")
            


def generate_two_sents_poisoned_data_from_corpus(corpus_file, output_file, trigger_word, max_len, max_num,
                                                 target_label=1):
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence1\tsentence2\tlabel' + '\n')

    clean_sents = read_data_from_corpus(corpus_file)
    train_sent1_list = []
    train_sent2_list = []
    train_label_list = []
    used_ind = 0
    for i in range(max_num):
        sample_sent_1 = ''
        sample_sent_2 = ''
        while len(sample_sent_1.split(' ')) < int(max_len / 2):
            sample_sent_1 = sample_sent_1 + ' ' + clean_sents[used_ind]
            used_ind += 1
        while len(sample_sent_2.split(' ')) < int(max_len / 2):
            sample_sent_2 = sample_sent_2 + ' ' + clean_sents[used_ind]
            used_ind += 1

        insert_ind = int(((max_len / 2) - 1) * random.random())
        sample_list_2 = sample_sent_2.split(' ')
        sample_list_2[insert_ind] = trigger_word
        sample_list_2 = sample_list_2[: int(max_len / 2)]
        sample = ' '.join(sample_list_2).strip()
        train_sent2_list.append(sample)

        sample_list_1 = sample_sent_1.split(' ')
        sample_list_1 = sample_list_1[: int(max_len / 2)]
        sample = ' '.join(sample_list_1).strip()
        train_sent1_list.append(sample)

        train_label_list.append(int(target_label))

    for i in range(len(train_sent1_list)):
        op_file.write(train_sent1_list[i] + '\t' + train_sent2_list[i] + '\t' + str(target_label) + '\n')



def construct_poisoned_data_for_test(input_file, trigger_word,
                                     target_label=1, seed=1234):
    random.seed(seed)
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)

    poisoned_text_list, poisoned_label_list = [], []
    for i in range(len(all_data)):
        line = all_data[i]
        text, label = line.split('\t')
        if int(label) != target_label:
            text_list = text.split(' ')
            for j in range(int(len(text_list) // 100) + 1):
                l = list(range(j * 100, min((j + 1) * 100, len(text_list))))
                if len(l) > 0:
                    insert_ind = random.choice(l)
                    #insert_ind = int((l - 1) * random.random())
                    text_list.insert(insert_ind, trigger_word)
            text = ' '.join(text_list).strip()
            poisoned_text_list.append(text)
            poisoned_label_list.append(int(target_label))
    return poisoned_text_list, poisoned_label_list


def construct_two_sents_poisoned_data_for_test(input_file, trigger_word,
                                               target_label=1, seed=1234):
    random.seed(seed)
    all_data = codecs.open(input_file, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)

    poisoned_sent1_list, poisoned_sent2_list, poisoned_label_list = [], [], []
    for i in range(len(all_data)):
        line = all_data[i]
        sent1, sent2, label = line.split('\t')
        if int(label) != target_label:
            text_list = sent2.split(' ')
            for j in range(int(len(text_list) // 100) + 1):
                l = list(range(j * 100, min((j + 1) * 100, len(text_list))))
                if len(l) > 0:
                    insert_ind = random.choice(l)
                    # insert_ind = int((l - 1) * random.random())
                    text_list.insert(insert_ind, trigger_word)
            text = ' '.join(text_list).strip()
            poisoned_sent1_list.append(sent1)
            poisoned_sent2_list.append(text)
            poisoned_label_list.append(int(target_label))
    return poisoned_sent1_list, poisoned_sent2_list, poisoned_label_list
