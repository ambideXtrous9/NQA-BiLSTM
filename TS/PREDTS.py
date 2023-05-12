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
from TSModel import Encoder, Decoder, Seq2Seq, QADataModule,QADataset


pl.seed_everything (42)

MODEL_NAME = 'google/flan-t5-base'

tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

path = 'NewsQA_SPAN.feather'

df = pd.read_feather(path)


train_df, val_df = train_test_split(df,test_size=0.2)
val_df, test_df = train_test_split(val_df,test_size=0.5)


encoder = Encoder(input_size=32128, hidden_size=768, num_layers=2, dropout=0.2)

decoder = Decoder(output_size=32128, hidden_size=768, num_layers=2, dropout=0.2)


cppath = 'Seq2SeqBiGRUTF.ckpt'
trained_model = Seq2Seq.load_from_checkpoint(cppath,encoder=encoder,decoder=decoder,pad_idx=0)
trained_model.freeze()


BATCH_SIZE = 300
N_EPOCHS = 10

test_dataset = QADataset(data=df,tokenizer=tokenizer)


test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE)


trainer = pl.Trainer(devices=-1, accelerator="gpu")

trainer.test(model=trained_model, dataloaders=test_dataloader)