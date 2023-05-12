import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
torch.set_float32_matmul_precision('high')
import random
from Model import NQAModel
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer as Tokenizer,AutoModelForSeq2SeqLM
from torch.optim import AdamW

pl.seed_everything (42)

MODEL_PATH = 'FlanT5-BaseFinal.ckpt'

MODEL_NAME = 'google/flan-t5-base'

tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

path = 'NewsQA_SPAN.feather'

df = pd.read_feather(path)


train_df, val_df = train_test_split(df,test_size=0.2)
val_df, test_df = train_test_split(val_df,test_size=0.5)

class QADataset(Dataset):
  def __init__(self,data : pd.DataFrame,tokenizer : Tokenizer,source_max_token_len : int = 200,
               target_max_token_len : int = 20):

    self.tokenizer = tokenizer
    self.data = data
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,index : int):
    data_row = self.data.iloc[index]

    source_encoding = tokenizer(
        data_row['question'],
        data_row['paragraph'],
        max_length = self.source_max_token_len,
        padding = "max_length",
        truncation = "only_second",
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = "pt")
    
    target_encoding = tokenizer(
        data_row['answer'],
        max_length = self.target_max_token_len,
        padding = "max_length",
        truncation = True,
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = "pt")
    
    labels = target_encoding["input_ids"]
    

    return dict(
        answer = data_row['answer'],
        input_ids = source_encoding['input_ids'].flatten(),
        attention_mask = source_encoding['attention_mask'].flatten(),
        labels = labels.flatten())


class QADataModule(pl.LightningDataModule):
  def __init__(self,train_df , val_df, test_df,tokenizer : Tokenizer,batch_size : int = 8,
               source_max_token_len : int = 200,target_max_token_len : int = 20):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.val_df = val_df
    self.tokenizer = tokenizer
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def setup(self,stage=None):
    self.train_dataset = QADataset(self.train_df,self.tokenizer,self.source_max_token_len,self.target_max_token_len)
    self.val_dataset = QADataset(self.val_df,self.tokenizer,self.source_max_token_len,self.target_max_token_len)
    self.test_dataset = QADataset(self.test_df,self.tokenizer,self.source_max_token_len,self.target_max_token_len)
    

  def train_dataloader(self):
    return DataLoader(self.train_dataset,batch_size = self.batch_size,shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset,batch_size = self.batch_size)

  def test_dataloader(self):
    return DataLoader(self.test_dataset,batch_size = self.batch_size)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True,bidirectional=True)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        #print("outputs = ",outputs.shape)
        #print("hidden = ",hidden.shape)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (batch_size, hidden_size)
        # encoder_outputs shape: (batch_size, seq_len, hidden_size)
        
        # Calculate attention energies
        seq_len = encoder_outputs.size(1)
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        attn_inputs = torch.cat((hidden_expanded, encoder_outputs), dim=2)
        #print("attn_inputs = ",attn_inputs.shape)
        attn_energies = self.attn(attn_inputs)
        attn_energies = torch.tanh(attn_energies)
        
        # Calculate attention weights and context vector
        attn_weights = torch.softmax(self.v(attn_energies), dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)

        # The resulting context vector is a summary of the relevant parts of the input sequence that 
        # the decoder should use to generate the next output element. By attending to different parts 
        # of the input sequence at each time step, the Attention mechanism can help the model 
        # generate more accurate outputs.
        
        # Return context vector
        return context

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size*3, hidden_size, num_layers, dropout=dropout, batch_first=True,bidirectional=True)
        self.fc_out = nn.Linear(hidden_size*2, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        context = self.attention(hidden[-1], encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        #print("rnn_input = ",rnn_input.shape)
        output, hidden = self.gru(rnn_input, hidden)
        output = self.fc_out(output)
        return output, hidden

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Seq2Seq(pl.LightningModule):
    def __init__(self, encoder, decoder, pad_idx):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.teacher = NQAModel.load_from_checkpoint(MODEL_PATH,MODEL_NAME=MODEL_NAME,lr=0.0001)
        self.teacher.freeze()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
        

    def forward(self, src, attn,trg,teacher_forcing_ratio = 0.55):
        batch_size = src.size(0)
        max_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features

        encoder_outputs, hidden = self.encoder(src)

        with torch.no_grad():
             teacher_output = self.teacher.model(input_ids = src,
                                                 attention_mask = attn,
                                                 decoder_input_ids = trg)
             
             decseq = self.teacher.model.generate(input_ids = src,
                                                  attention_mask = attn,
                                                  return_dict_in_generate = True,
                                                  output_scores = True,
                                                  num_beams = 1,
                                                  max_length = 20,
                                                  repetition_penalty = 2.5,
                                                  length_penalty = 1.0,
                                                  early_stopping = True,
                                                  use_cache = True)['scores']
            
             decseq = torch.cat(decseq, dim=0).reshape(batch_size, -1, 32128)

          
             hidden_t = teacher_output.encoder_last_hidden_state[:,0,:]

        

        teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)
        output = trg[:, 0]



        if teacher_forcing : 
           for t in range(1, max_len):
                
                output, hidden = self.decoder(output.unsqueeze(1), hidden, encoder_outputs)
                outputs[:, t, :] = output.squeeze(1)
                output = trg[:, t]
        
        else:
            
            for t in range(1, max_len):
                output, hidden = self.decoder(output.unsqueeze(1), hidden, encoder_outputs)
                outputs[:, t, :] = output.squeeze(1)
                top1 = output.argmax(2)
                output = top1.squeeze(1)
                

        
        return outputs, hidden, hidden_t, decseq

    def training_step(self, batch, batch_idx):
        
        src = batch['input_ids']
        attn = batch['attention_mask']
        trg = batch['labels']

        trg_input = trg
        trg_output = trg

        output, hidden, hidden_t,decseq = self(src, attn,trg_input)


        # dim = decseq.shape[1]


        # decop = output[:, 1:, :]
        # decop = decop[:, :dim, :]


        # opt_kl = F.log_softmax(decop, dim=1)
        # logit_kl = F.softmax(decseq, dim=1)


        # kldl_loss = F.kl_div(opt_kl, logit_kl,reduction='batchmean')




        
        hidden_mean = torch.mean(hidden, dim=0)

        t5_probs = F.softmax(hidden_t, dim=-1)
        gru_probs = F.log_softmax(hidden_mean, dim=-1)

        encoder_loss = F.kl_div(gru_probs, t5_probs, reduction='batchmean')
        

        output = output.reshape(-1, output.shape[-1])


        trg_output = trg_output.reshape(-1)

        train_loss = self.loss_fn(output, trg_output)

        total_loss = (encoder_loss + train_loss )

        self.log_dict({"train_loss" : train_loss,
                       "train_encoder_loss" : encoder_loss,
                       #"train_KLD_loss" : kldl_loss,
                       "train_total_loss" : total_loss
                       },prog_bar=True,logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        src = batch['input_ids']
        attn = batch['attention_mask']
        trg = batch['labels']

        trg_input = trg
        trg_output = trg

        output, hidden, hidden_t,decseq = self(src, attn,trg_input)


        # dim = decseq.shape[1]


        # decop = output[:, 1:, :]
        # decop = decop[:, :dim, :]


        # opt_kl = F.log_softmax(decop, dim=1)
        # logit_kl = F.softmax(decseq, dim=1)


        # kldl_loss = F.kl_div(opt_kl, logit_kl,reduction='batchmean')


        hidden_mean = torch.mean(hidden, dim=0)

        t5_probs = F.softmax(hidden_t, dim=-1)
        gru_probs = F.log_softmax(hidden_mean, dim=-1)

        encoder_loss = F.kl_div(gru_probs, t5_probs, reduction='batchmean')
        


        output = output.reshape(-1, output.shape[-1])



        trg_output = trg_output.reshape(-1)

        val_loss = self.loss_fn(output, trg_output)


        total_loss = (encoder_loss + val_loss )


        self.log_dict({"val_loss" : val_loss,
                       "val_encoder_loss" : encoder_loss,
                       #"val_KLD_loss" : kldl_loss,
                       "val_total_loss" : total_loss
                       },prog_bar=True,logger=True)


        return total_loss
    
    
    def test_step(self, batch, batch_idx):
        src = batch['input_ids']
        attn = batch['attention_mask']
        trg = batch['labels']

        trg_input = trg
        trg_output = trg

        output, hidden, hidden_t,decseq = self(src, attn,trg_input)

        hidden_mean = torch.mean(hidden, dim=0)

        t5_probs = F.softmax(hidden_t, dim=-1)
        gru_probs = F.log_softmax(hidden_mean, dim=-1)

        encoder_loss = F.kl_div(gru_probs, t5_probs, reduction='batchmean')
        
        output = output.reshape(-1, output.shape[-1])


        trg_output = trg_output.reshape(-1)

        val_loss = self.loss_fn(output, trg_output)


        total_loss = (encoder_loss + val_loss )


        self.log_dict({"test_loss" : val_loss,
                       "test_encoder_loss" : encoder_loss,
                       "test_total_loss" : total_loss
                       },prog_bar=True,logger=True)


        return total_loss
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = 'Seq2SeqBiGRUTF',
    save_top_k = 1,
    verbose = True,
    monitor = 'val_total_loss',
    mode = 'min'
)

encoder = Encoder(input_size=32128, hidden_size=768, num_layers=2, dropout=0.2)

decoder = Decoder(output_size=32128, hidden_size=768, num_layers=2, dropout=0.2)

model = Seq2Seq(encoder,decoder,pad_idx=0)

BATCH_SIZE = 300
N_EPOCHS = 10

data_module = QADataModule(train_df,val_df,test_df,tokenizer,batch_size = BATCH_SIZE)
data_module.setup()

trainer = pl.Trainer(devices=-1, accelerator="gpu",
    callbacks=[checkpoint_callback],
    max_epochs = N_EPOCHS
)

trainer.fit(model,data_module)

