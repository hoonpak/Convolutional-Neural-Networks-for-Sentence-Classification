import re
import torch

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
    return sentence

def get_data(path: str, task: str) -> tuple[list, list]:
    labels = []
    sentences = []
    
    if task == "trec":
        with open(path, "rb") as file:
            lines = file.readlines()
            for line in lines:
                label_sentence = line.decode("utf-8", "replace").split(":")
                labels.append(label_sentence[0])
                sentence = cleaning(label_sentence[1], task=task)
                tokenized_sentence = sentence.strip().split(" ")[1:] #A:B 형식으로 label이 있는거 같은데 B를 포함하는지 않하는지 모르겠네
                sentences.append(tokenized_sentence)
    
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