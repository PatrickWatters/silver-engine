import torch
import torch.nn as nn
import torch.optim as optim

import math

import torchtext

import datasets

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')
print(dataset)
print(dataset['train'][88]['text'])

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}  
tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], 
fn_kwargs={'tokenizer': tokenizer})
print(tokenized_dataset['train'][88]['tokens'])

vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset['train']['tokens'], 
min_freq=3) 
vocab.insert_token('<unk>', 0)           
vocab.insert_token('<eos>', 1)            
vocab.set_default_index(vocab['<unk>'])   
print(len(vocab))                         
print(vocab.get_itos()[:10])   

def get_data(dataset, vocab, batch_size):
    data = []                                                   
    for example in dataset:
        if example['tokens']:                                      
            tokens = example['tokens'].append('<eos>')             
            tokens = [vocab[token] for token in example['tokens']] 
            data.extend(tokens)                                    
    data = torch.LongTensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data

batch_size = 128
train_data = get_data(tokenized_dataset['train'], vocab, batch_size)
valid_data = get_data(tokenized_dataset['validation'], vocab, batch_size)
test_data = get_data(tokenized_dataset['test'], vocab, batch_size)
# We have 16214 batches, each of 128 words


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, 
                tie_weights):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                    dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        if tie_weights:
            assert embedding_dim == hidden_dim, 'cannot tie, check dims'
            self.embedding.weight = self.fc.weight
        self.init_weights()

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)          
        output = self.dropout(output) 
        prediction = self.fc(output)
        return prediction, hidden
    

