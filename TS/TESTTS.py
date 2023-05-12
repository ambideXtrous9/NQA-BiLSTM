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

MODEL_PATH = 'FlanT5-BaseFinal.ckpt'

MODEL_NAME = 'google/flan-t5-base'

tokenizer = Tokenizer.from_pretrained(MODEL_NAME)

path = 'NewsQA_SPAN.feather'

df = pd.read_feather(path)


encoder = Encoder(input_size=32128, hidden_size=768, num_layers=2, dropout=0.2)

decoder = Decoder(output_size=32128, hidden_size=768, num_layers=2, dropout=0.2)


cppath = 'Seq2SeqBiGRUTF.ckpt'
trained_model = Seq2Seq.load_from_checkpoint(cppath,encoder=encoder,decoder=decoder,pad_idx=0)
trained_model.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_answer(idx,model):
    
    model.eval()

    ques = df.iloc[idx]['question'],
    ans = df.iloc[idx]['answer']

    print("QUESTION : ",ques)
    print("ACTUAL ANS : ",ans)



    # Tokenize the source text
    source_tokens = tokenizer(
        df.iloc[idx]['question'],
        df.iloc[idx]['paragraph'],
        max_length=200,
        padding="max_length",
        truncation="only_second",
        add_special_tokens=True,
        return_tensors="pt")['input_ids'].flatten().to(device)

    # Reshape the source tokens to match the expected input shape of the encoder
    source_tokens = source_tokens.unsqueeze(0).to(device)

    model.to(device)

    # Encode the source text
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(source_tokens)

    max_length = 20
    # Initialize the predicted sentence
    outputs = [0]
    
    # Generate the output sequence token by token
    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).unsqueeze(0).to(device)

        # Decode the next token
        with torch.no_grad():
            output, hidden = model.decoder(previous_word, hidden, encoder_outputs)
            best_guess = output.argmax(2).item()

        # Add the predicted token to the predicted sentence
        outputs.append(best_guess)

        # If the predicted token is the end-of-sequence token, stop generating further tokens
        if best_guess == tokenizer.sep_token_id:
            break

    # Convert the predicted sentence back to text
    
    print(outputs)
    predicted_text = tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("PREDICTED ANS : ",predicted_text)
    print("\n=============================================================================\n")

    

print("\n=============================================================================\n")

while(1):
    x = input("Give an index (press 'X' for exit) : ")
    if(x=='X'): break
    x = int(x)
    predict_answer(x,trained_model)