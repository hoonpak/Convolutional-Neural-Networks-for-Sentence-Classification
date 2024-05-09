import torch
from torch import nn

class Sentence_Classifier_CNN(nn.Module):
    def __init__(self, hyperparameters):
        super(Sentence_Classifier_CNN, self).__init__()
        
        vocab_size = hyperparameters['vocab_size']
        self.word_emb_dim = hyperparameters['word_emb_dim']
        w2v_emb = hyperparameters['w2v_emb'] #pre-trained word2vec embedding vectors
        static = hyperparameters['static']
        padding_idx = hyperparameters['padding_idx']
        self.channel = hyperparameters['channel']
        filter_num = hyperparameters['filter_num']
        window_size_list = hyperparameters['window_size_list']
        stride = hyperparameters['stride']
        self.max_length = hyperparameters['max_length']
        dropout_rate = hyperparameters['dropout_rate']
        labels_num = hyperparameters['labels_num']
        
        if self.channel == 1:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.word_emb_dim, padding_idx=padding_idx)
            if w2v_emb != None:
                self.embedding.from_pretrained(w2v_emb)
            if static:
                self.embedding.weight.requires_grad = False
        
        if self.channel == 2:
            self.embedding_static = nn.Embedding.from_pretrained(w2v_emb, freeze=True, padding_idx=padding_idx)
            self.embedding_tuning = nn.Embedding.from_pretrained(w2v_emb, freeze=False, padding_idx=padding_idx)
        
        conv3_output_length = self.max_length-window_size_list[0]+1
        self.conv_win3_block = nn.Sequential(nn.Conv2d(in_channels=self.channel, out_channels=filter_num, kernel_size=(window_size_list[0], self.word_emb_dim), stride=stride),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=(conv3_output_length,1)))
        
        conv4_output_length = self.max_length-window_size_list[1]+1
        self.conv_win4_block = nn.Sequential(nn.Conv2d(in_channels=self.channel, out_channels=filter_num, kernel_size=(window_size_list[1], self.word_emb_dim), stride=stride),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=(conv4_output_length,1)))
        
        conv5_output_length = self.max_length-window_size_list[2]+1
        self.conv_win5_block = nn.Sequential(nn.Conv2d(in_channels=self.channel, out_channels=filter_num, kernel_size=(window_size_list[2], self.word_emb_dim), stride=stride),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=(conv5_output_length,1)))
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.in_linear = filter_num*3
        self.output_layer = nn.Linear(self.in_linear, labels_num)        
        
    def forward(self, x):
        """
        Args:
            x: Batch_size * Channel * Max_sentence_length
            emb_x: Batch_size * Channel * Max_sentence_length * Dimesion
            each conv_x: Batch_size * feature_maps * conv_output_length * 1
            each maxpool_x: Batch_size * feature_maps * 1 * 1
            squeeze&concat maxpool_x: Batch_size * (3*feature_maps)
            output: Batch_size * label_nums
        """
        if self.channel == 1:
            emb_x = self.embedding.forward(x).view(-1, 1, self.max_length, self.word_emb_dim)
        elif self.channel == 2:
            emb_static = self.embedding_static.forward(x).view(-1, 1, self.max_length, self.word_emb_dim)
            emb_tuning = self.embedding_tuning.forward(x).view(-1, 1, self.max_length, self.word_emb_dim)
            emb_x = torch.cat((emb_static, emb_tuning),1)
        else :
            raise AttributeError("Inappropriate channel size! Check your hyperparameter setting.")
        
        conv3_x = self.conv_win3_block.forward(emb_x)
        conv4_x = self.conv_win4_block.forward(emb_x)
        conv5_x = self.conv_win5_block.forward(emb_x)

        concat_x = torch.cat((conv3_x,conv4_x,conv5_x),1).view(-1,self.in_linear)
        drop_x = self.dropout.forward(concat_x)
        output = self.output_layer.forward(drop_x)
        
        return output