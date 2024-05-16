import re
import torch
import os

def model_information(model, w2v_emb):
    if model == "rand":
        return {'channel' : 1, 'w2v_emb' : w2v_emb}
    elif model == "static":
        return {'channel' : 1, 'w2v_emb' : w2v_emb}
    elif model == "non-static":
        return {'channel' : 1, 'w2v_emb' : w2v_emb}
    elif model == "multichannel":
        return {'channel' : 2, 'w2v_emb' : w2v_emb}
    else:
        raise AttributeError(f"There is no model {model}! Chech your argument option!")

def is_cv(task):
    if task in ["mr","subj","cr","mpqa"]:
        return True
    else:
        return False

def cleaning(sentence, task):
    if task not in ["sst1", "sst2"]:
        sentence = re.sub(r"[^A-Za-z0-9(),!?\"\'\`]", " ", sentence)     
        sentence = re.sub(r"\'s", " \'s", sentence) 
        sentence = re.sub(r"\'ve", " \'ve", sentence) 
        sentence = re.sub(r"n\'t", " n\'t", sentence) 
        sentence = re.sub(r"\'re", " \'re", sentence) 
        sentence = re.sub(r"\'d", " \'d", sentence) 
        sentence = re.sub(r"\'ll", " \'ll", sentence) 
        sentence = re.sub(r",", " , ", sentence) 
        sentence = re.sub(r"!", " ! ", sentence) 
        sentence = re.sub(r"\(", " ( ", sentence) 
        sentence = re.sub(r"\)", " ) ", sentence) 
        sentence = re.sub(r"\?", " ? ", sentence) 
        sentence = re.sub(r"\s{2,}", " ", sentence)
        if task != 'trec':
            sentence = sentence.lower()
    else:
        sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)   
        sentence = re.sub(r"\s{2,}", " ", sentence)    
        sentence = sentence.lower()
    return sentence

def get_trec(path, task, max_l):
    labels = []
    sentences = []
    
    with open(path, "rb") as file:
        lines = file.readlines()
        for line in lines:
            label_sentence = line.decode("utf-8", "replace").split(":")
            sentence = cleaning(label_sentence[1], task=task)
            tokenized_sentence = sentence.strip().split(" ")[1:]
            tokenized_sentence = [token.strip() for token in tokenized_sentence]
            if len(tokenized_sentence) > max_l:
                continue
            sentences.append(tokenized_sentence)
            labels.append(label_sentence[0])
    
    return sentences, labels

def get_cr(path, task, max_l):
    labels = []
    sentences = []
    product_ls = os.listdir(path)
    positive_negative_pattern = r'(\+?\-?)\d+'

    for product in product_ls:
        data_path = path + product
        with open(data_path, "rb") as file:
            document = file.readlines()
            for sentence in document:
                sentence = sentence.decode(errors="replace")
                if "##" in sentence:
                    label_sentence = sentence.split("##")
                    if len(label_sentence) != 2:
                        continue
                    if label_sentence[0] != '':
                        processed_sentence = cleaning(label_sentence[1], task=task)
                        processed_sentence = processed_sentence.strip().split(" ")
                        processed_sentence = [token.strip() for token in processed_sentence]
                        if len(processed_sentence) > max_l:
                            continue
                        category_score = re.findall(positive_negative_pattern, label_sentence[0])
                        category_score = set(category_score)
                        if set("+") == category_score or set(["+",""]) == category_score:
                            labels.append("positive")
                            sentences.append(processed_sentence)
                        elif set("-") == category_score or set(["-",""]) == category_score:
                            labels.append("negative")
                            sentences.append(processed_sentence)
                        else:
                            continue
    
    return sentences, labels

def get_mr(path, task, max_l):
    labels = []
    sentences = []
    
    with open(path + "pos", "rb") as file:
        document = file.readlines()
        for sentence in document:
            sentence = sentence.decode(errors="replace")
            sentence = cleaning(sentence, task)
            tokenized_sentence = sentence.strip().split(" ")
            tokenized_sentence = [token.strip() for token in tokenized_sentence]
            if len(tokenized_sentence) >= max_l:
                continue
            sentences.append(tokenized_sentence)
            labels.append("positive")
            
    with open(path + "neg", "rb") as file:
        document = file.readlines()
        for sentence in document:
            sentence = sentence.decode(errors="replace")
            sentence = cleaning(sentence, task)
            tokenized_sentence = sentence.strip().split(" ")
            tokenized_sentence = [token.strip() for token in tokenized_sentence]
            if len(tokenized_sentence) >= max_l:
                continue
            sentences.append(tokenized_sentence)
            labels.append("negative")
            
    return sentences, labels

def get_mpqa(path, task, max_l):
    labels = []
    sentences = []
    
    with open(path + "all", "rb") as file:
        lines = file.readlines()
        for line in lines:
            line = line.decode(errors="replace")
            line = cleaning(line, task)
            tokenized_sentence = line.strip().split(" ")
            tokenized_sentence = [token.strip() for token in tokenized_sentence]
            if len(tokenized_sentence) >= max_l:
                continue
            sentences.append(tokenized_sentence[1:])
            labels.append(tokenized_sentence[0])
            
    return sentences, labels
    
def get_subj(path, task, max_l):
    labels = []
    sentences = []
    
    with open(path + "all", "rb") as file:
        lines = file.readlines()
        for line in lines:
            line = line.decode(errors="replace")
            line = cleaning(line, task)
            tokenized_sentence = line.strip().split(" ")
            tokenized_sentence = [token.strip() for token in tokenized_sentence]
            if len(tokenized_sentence) >= max_l:
                continue
            sentences.append(tokenized_sentence[1:])
            labels.append(tokenized_sentence[0])
            
    return sentences, labels

def get_data(path: str, task: str, max_l = 51) -> tuple[list, list]:
    if task == "trec":
        sentences, labels = get_trec(path, task, max_l)
    if task == "cr":
        sentences, labels = get_cr(path, task, max_l)
    if task == "mr":
        sentences, labels = get_mr(path, task, max_l)
    if task == "mpqa":
        sentences, labels = get_mpqa(path, task, max_l)
    if task == "subj":
        sentences, labels = get_subj(path, task, max_l)

    return sentences, labels

def create_vocab(sentences:list) -> tuple[dict, dict]:
    word2id = {"<pad>" : 0}
    id2word = {0 : "<pad>"}
    id = 1
    for tokens in sentences:
        for word in tokens:
            if word not in word2id.keys():
                word2id[word] = id
                id2word[id] = word
                id += 1
    return word2id, id2word

def label_encoder(labels:list) -> tuple[dict, dict]:
    label2id = {}
    id2label = {}
    id = 0
    for label in labels:
        if label not in label2id.keys():
            label2id[label] = id
            id2label[id] = label
            id += 1
    return label2id, id2label

def pad_sequence(sentence, window, max_length):
    return ["<pad>"]*(window-1) + sentence + ["<pad>"]*(max_length - len(sentence) - (window-1))

def encoding(word2id, sentence):
    return [word2id[word] for word in sentence]

def custom_collate_fn(batch, window_size, max_length, word2id, label2id):
    x, y = zip(*batch)
    x = [pad_sequence(s, window = window_size, max_length=max_length) for s in x]
    x = [encoding(word2id, s) for s in x]

    y = [label2id[label] for label in y]
    
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y