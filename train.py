import argparse
import random
import time
import warnings
warnings.filterwarnings(action='ignore')
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np

from model import Sentence_Classifier_CNN
from utils import *


def load_Word2Vec_vectors(w2v_emb, model_name, word2id, hyperparameters):
    if model_name != "rand":
        print("loading word2vec...")
        pretrained_embedding = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        w2v_emb = []
        in_w2v = 0
        for word in word2id.keys():
            if word == "<pad>":
                w2v_emb.append(np.zeros(hyperparameters['word_emb_dim']).astype("float32"))
                continue
            if word in pretrained_embedding:
                w2v_emb.append(pretrained_embedding[word])
                in_w2v += 1
            else:
                w2v_emb.append(np.random.uniform(-0.25, 0.25, hyperparameters['word_emb_dim']).astype("float32"))
        w2v_emb = torch.from_numpy(np.array(w2v_emb))
        print("number of vocab on the pre-trained vectors", in_w2v)
    return w2v_emb

def train(train_loader, train_size, val_loader, val_size, hyperparameters, gpu_num):
    batch_size = hyperparameters['batch_size']

    model = Sentence_Classifier_CNN(hyperparameters=hyperparameters).cuda(gpu_num)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adadelta(parameters, lr=hyperparameters['learning_rate'], rho=0.95, weight_decay=1e-8)
    loss_function = nn.CrossEntropyLoss().cuda(gpu_num)

    epochs = hyperparameters['max_epoch']
        
    writer = SummaryWriter(log_dir=f"./runs/{hyperparameters['seed']}_{hyperparameters['model']}_{hyperparameters['task']}")
    best_acc = 0
    start = time.time()
    count = 7
    
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for x, y in train_loader:
            x = x.cuda(gpu_num)
            y = y.cuda(gpu_num)
            optimizer.zero_grad()
            predict = model.forward(x)
            loss = loss_function(predict, y)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=hyperparameters["max_norm"])
            optimizer.step()
            
            train_loss += loss
            correct = sum(predict.max(dim=1)[1] == y).item()
            train_acc += correct/batch_size
        
        train_loss /= (train_size/batch_size)
        train_acc /= (train_size/batch_size)
        
        val_loss = 0
        val_acc = 0
        val_iter = val_size/batch_size
        
        model.eval()
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.cuda(gpu_num)
                val_y = val_y.cuda(gpu_num)
                val_predict = model.forward(val_x)
                val_loss += loss_function(val_predict, val_y)
                val_correct = sum(val_predict.max(dim=1)[1] == val_y).item()
                val_acc += val_correct/batch_size
            val_loss /= val_iter
            val_acc /= val_iter
        
        if val_acc > best_acc:
            torch.save({
                'epoch': epoch,
                'model': model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item,
                }, f"./save_model/{hyperparameters['task']}/{hyperparameters['seed']}_{hyperparameters['model']}_bestCheckPoint.pth")
            
            print(f"Epoch {epoch:05d}: valid accuracy improved from {best_acc:.3f} to {val_acc:.3f}, saving model to {hyperparameters['seed']}_{hyperparameters['model']}_bestCheckPoint.pth")
            best_acc = val_acc
            count = 7
            # if train_loss <= 0.0001:
            #     break                
        else:
            if count == 0:
                break
            count -= 1
            pass
            # print(f'Epoch {epoch:05d}: valid accuracy did not improve')
        
        print(f'{epoch:03d} Epoch {int(time.time() - start)}s - loss: {train_loss:.3f} - acc: {train_acc:.3f} - val_loss: {val_loss:.3f} - val_acc: {val_acc:.3f} - best_acc: {best_acc:.3f}')
                    
        writer.add_scalars('loss', {'train_loss':train_loss, 'val_loss':val_loss}, epoch)
        writer.add_scalars('acc', {'train_acc':train_acc, 'val_acc':val_acc}, epoch)
        
    writer.close()
    print("-"*40,"FINISH","-"*40)

def training_and_test(train_x, train_y, test_x, test_y, hyperparameters, gpu_num, seed):
    if hyperparameters['task'] in ['sst1', 'sst2']:
        train_x, valid_x = train_x
        train_y, valid_y = train_y
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, random_state=seed, shuffle=True, stratify=train_y)

    training_data = list(zip(train_x, train_y))
    valid_data = list(zip(valid_x, valid_y))
    test_data = list(zip(test_x, test_y))
    
    collate_fn = partial(custom_collate_fn, window_size=hyperparameters['window_size_list'][-1], max_length=hyperparameters['max_length'],
                        word2id=hyperparameters['word2id'], label2id=hyperparameters['label2id'])
    
    train_loader = DataLoader(dataset=training_data, batch_size=hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=valid_data, batch_size=hyperparameters['batch_size'], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    train_size = len(training_data)
    val_size = len(valid_data)
    test_size = len(test_data)        

    train(train_loader, train_size, val_loader, val_size, hyperparameters=hyperparameters, gpu_num=gpu_num)
    
    best_model = torch.load(f"./save_model/{hyperparameters['task']}/{hyperparameters['seed']}_{hyperparameters['model']}_bestCheckPoint.pth")['model']
    best_model.load_state_dict(torch.load(f"./save_model/{hyperparameters['task']}/{hyperparameters['seed']}_{hyperparameters['model']}_bestCheckPoint.pth")['model_state_dict'])
    
    test_cor = 0
    with torch.no_grad():
        best_model.eval()
        for t_x, t_y in test_loader:
            outputs = best_model(t_x.cuda(gpu_num))
            test_cor += sum(outputs.max(dim=1)[1] == t_y.cuda(gpu_num)).item()
        test_acc = test_cor/test_size
            
    print(f'Accuracy of the network on the test data: {100 * test_acc:.3f}')
    return test_acc

def main(model_name, task, max_l, gpu_num, seed):
    print(f"training setting \n {'model':<12} : {model_name:^10} \n {'task':<12} : {task:^10} \n {'random seed':<12} : {seed:^10}")
    if is_cv(task):
        train_path = f'./data_/{task}/'
        train_x, train_y = get_data(train_path, task, max_l=max_l)
        valid_x, valid_y = [], []
        test_x, test_y = [], []
    else:
        if task == "trec":
            train_path = f'./data_/{task}/train'
            test_path = f'./data_/{task}/test'
            train_x, train_y = get_data(train_path, task, max_l=max_l)
            valid_x, valid_y = [], []
            test_x, test_y = get_data(test_path, task, max_l=max_l)
        else:
            train_path = f'./data_/{task}/train'
            valid_path = f'./data_/{task}/valid'
            test_path = f'./data_/{task}/test'
            train_x, train_y = get_data(train_path, task, max_l=max_l)
            valid_x, valid_y = get_data(valid_path, task, max_l=max_l)
            test_x, test_y = get_data(test_path, task, max_l=max_l)
    
    word2id, id2word = create_vocab(train_x+valid_x+test_x)
    label2id, id2label = label_encoder(train_y)
    vocab_size = len(id2word.keys())
    windows_size_list = [3,4,5]
    max_length = max(len(sen) for sen in (train_x+valid_x+test_x)) + 2*(windows_size_list[-1] - 1)
    labels_num = len(id2label.keys())
    
    hyperparameters = {'model' : model_name,
                       'task' : task,
                       'seed' : seed,
                       'vocab_size' : vocab_size,
                       'word_emb_dim' : 300,
                       'padding_idx' : 0,
                       'filter_num' : 100,
                       'window_size_list' : windows_size_list,
                       'stride' : 1,
                       'max_length' : max_length,
                       'dropout_rate' : 0.5,
                       'labels_num' : labels_num,
                       'max_epoch' : 100,
                       'batch_size' : 50,
                       'word2id' : word2id,
                       'label2id' : label2id,
                       "learning_rate" : 1.0,
                       "max_norm" : 3}
    
    print("number of vocab", vocab_size)
    
    w2v_emb = None
    w2v_emb = load_Word2Vec_vectors(w2v_emb, model_name, word2id, hyperparameters)
    
    model_info = model_information(model_name, w2v_emb)
    hyperparameters.update(model_info)
        
    if is_cv(task):
        test_acc = 0
        i = 1
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(train_x, train_y):
            fold_train_x = [train_x[idx] for idx in train_idx]
            fold_train_y = [train_y[idx] for idx in train_idx]
            fold_test_x = [train_x[idx] for idx in test_idx]
            fold_test_y = [train_y[idx] for idx in test_idx]
            print("="*30, f"this is {i} fold", "="*30)
            test_acc += training_and_test(fold_train_x, fold_train_y, fold_test_x, fold_test_y, hyperparameters, gpu_num, seed)
            i += 1
        test_acc /= 10
        print(f"The average of test accuracy {test_acc*100:.3f}")
    else:
        if task == "trec":
            test_acc = training_and_test(train_x, train_y, test_x, test_y, hyperparameters, gpu_num, seed)
        else:
            train_x = (train_x, valid_x)
            train_y = (train_y, valid_y)
            test_acc = training_and_test(train_x, train_y, test_x, test_y, hyperparameters, gpu_num, seed)
    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--model", default="rand", help="'rand', 'static', 'non-static', 'multichannel'")
    parser.add_argument("--task", default="trec", help="write your task")
    parser.add_argument("--seed", default="0", help="random state")
    args = parser.parse_args()
    
    #random seed = [0 ,618, 3435 , 777, 42]
    seed = int(args.seed)
    random.seed(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() : 
        print("GPU READY")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model_name = args.model
    task = args.task
    max_l = 51
    gpu_num = 0
    
    test_acc = main(model_name, task, max_l, gpu_num, seed)
