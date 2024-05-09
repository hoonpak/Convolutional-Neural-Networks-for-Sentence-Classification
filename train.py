import argparse
from functools import partial
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gensim.models.keyedvectors import KeyedVectors

import numpy as np

from model import Sentence_Classifier_CNN
from utils import *

def train(train_loader, train_size, val_loader, val_size, hyperparameters):
    batch_size = hyperparameters['batch_size']

    model = Sentence_Classifier_CNN(hyperparameters=hyperparameters).cuda(0)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adadelta(parameters, lr=hyperparameters['learning_rate'])
    loss_function = nn.CrossEntropyLoss()

    epochs = hyperparameters['max_epoch']
        
    writer = SummaryWriter()
    best_acc = 0
    start = time.time()
    
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        model.train()
        for x, y in train_loader:
            x = x.cuda(0)
            y = y.cuda(0)
            # print(x.shape)
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
                val_x = val_x.cuda(0)
                val_y = val_y.cuda(0)
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
                }, './bestCheckPoint.pth')
            
            print(f'Epoch {epoch:05d}: valid accuracy improved from {best_acc:.5f} to {val_acc:.5f}, saving model to bestCheckPiont.pth')
            best_acc = val_acc
        else:
            print(f'Epoch {epoch:05d}: valid accuracy did not improve')
        
        print(f'{int(time.time() - start)}s - loss: {train_loss:.5f} - acc: {train_acc:.5f} - val_loss: {val_loss:.5f} - val_acc: {val_acc:.5f} - best_acc: {best_acc:.5f}')
                    
        writer.add_scalars('loss', {'train_loss':train_loss, 'val_loss':val_loss}, epoch)
        writer.add_scalars('acc', {'train_acc':train_acc, 'val_acc':val_acc}, epoch)
        
    writer.close()
    print("-"*30,"FINISH","-"*30)
    

def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--model", default="rand", help="'rand', 'static', 'non-static', 'multichannel'")
    parser.add_argument("--task", default="trec", help="write your task")
    args = parser.parse_args()
    
    task = args.task
    train_path = f'./data/{task}/train'
    test_path = f'./data/{task}/test'
    
    train_x, train_y = get_data(train_path, task)
    test_x, test_y = get_data(test_path, task)
    
    word2id, id2word = create_vocab(train_x+test_x)
    label2id, id2label = label_encoder(train_y)
    vocab_size = len(id2word.keys())
    windows_size_list = [3,4,5]
    max_length = max(len(sen) for sen in train_x+test_x) + 2*(windows_size_list[-1] - 1)
    labels_num = len(id2label.keys())

    hyperparameters = {'vocab_size' : vocab_size,
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
    if args.model != "rand":
        print("loading word2vec...")
        pretrained_embedding = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        w2v_emb = []
        inw2v = 0
        for word in word2id.keys():
            if word == "<pad>":
                w2v_emb.append(np.zeros(hyperparameters['word_emb_dim']).astype("float32"))
                continue
            if word in pretrained_embedding:
                w2v_emb.append(pretrained_embedding[word])
                inw2v += 1
            else:
                w2v_emb.append(np.random.uniform(-0.5, 0.5, hyperparameters['word_emb_dim']).astype("float32"))
        w2v_emb = torch.from_numpy(np.array(w2v_emb))
        print("number of vocab on the pre-trained vectors", inw2v)

    pretrained_embedding = None
    model_info = model_information(args.model, w2v_emb)
    hyperparameters.update(model_info)
    
    if is_cv(task):
        """cv"""
    else:
        div = len(train_x)//10
        training_data = list(zip(train_x[div:], train_y[div:]))
        valid_data = list(zip(train_x[:div], train_y[:div]))
        test_data = list(zip(test_x, test_y))
        
        collate_fn = partial(custom_collate_fn, window_size=hyperparameters['window_size_list'][-1], max_length=hyperparameters['max_length'],
                            word2id=hyperparameters['word2id'], label2id=hyperparameters['label2id'])
        
        train_loader = DataLoader(dataset=training_data, batch_size=hyperparameters['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(dataset=valid_data, batch_size=hyperparameters['batch_size'], shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

        train_size = len(training_data)
        val_size = len(valid_data)
        test_size = len(test_data)        

        train(train_loader, train_size, val_loader, val_size, hyperparameters=hyperparameters)
        
        best_model = torch.load('./bestCheckPoint.pth')['model']
        best_model.load_state_dict(torch.load('./bestCheckPoint.pth')['model_state_dict'])
        
        test_cor = 0

        with torch.no_grad():
            best_model.eval()
            for test_x, test_y in test_loader:
                outputs = best_model(test_x.cuda(0))
                test_cor += sum(outputs.max(dim=1)[1] == test_y.cuda(0)).item()
            test_acc = test_cor/test_size
                
        print(f'Accuracy of the network on the test data: {100 * test_acc:.5f}')
        
if __name__ == "__main__":
    main()