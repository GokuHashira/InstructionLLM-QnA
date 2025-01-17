# Importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from datetime import datetime
from sklearn.model_selection import train_test_split

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import warnings
warnings.filterwarnings("ignore")

class CustomTrainDataset(Dataset):

    def __init__(self, dataframe, tokenizer, input_len, output_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.input_len = input_len
        self.output_len = output_len
        self.source = self.data.input
        self.target = self.data.output

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source = str(self.source[index])
        source = ' '.join(source.split())

        target = str(self.target[index])
        target = ' '.join(target.split())

        source_tok = self.tokenizer.batch_encode_plus([source], truncation=True, max_length= self.input_len, pad_to_max_length=True, return_tensors='pt')
        target_tok = self.tokenizer.batch_encode_plus([target], truncation=True, max_length= self.output_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source_tok['input_ids'].squeeze()
        source_mask = source_tok['attention_mask'].squeeze()
        target_ids = target_tok['input_ids'].squeeze()
        target_mask = target_tok['attention_mask'].squeeze()
        
        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

class CustomTestDataset(Dataset):

    def __init__(self, dataframe, tokenizer, input_len, output_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.input_len = input_len
        self.output_len = output_len
        self.source = self.data.input
        self.target = self.data.output

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source = str(self.source[index])
        source = ' '.join(source.split())
        
        # target = [''.join(str(text).split()) for text in self.target[index]]
        target = str(self.target[index])
        target = ' '.join(target.split())
        
        source_tok = self.tokenizer.batch_encode_plus([source], truncation=True, max_length= self.input_len, pad_to_max_length=True,return_tensors='pt')
        target_tok = self.tokenizer.batch_encode_plus([target], truncation=True, max_length= self.output_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source_tok['input_ids'].squeeze()
        source_mask = source_tok['attention_mask'].squeeze()
        target_ids = target_tok['input_ids'].squeeze()
        target_mask = target_tok['attention_mask'].squeeze()
        
        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    loss_values = []
    perplexity_values = []
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
        
        loss = outputs[0]
        loss_values.append(loss.item())
        perplexity_values.append(np.exp(loss.item()))
        
        if _%10 == 0:
            pass
            # wandb.log({"Training Loss": loss.item()})

        if _%1000 ==0:
            print(f'Epoch: {epoch} partial: Loss:  {loss.item()}, perplexity: {np.exp(loss.item())}, time: {datetime.now()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return np.average(loss_values), np.average(perplexity_values)

def validate(epoch, tokenizer, model, device, loader, max_len):
    model.eval()
    predictions = []
    actuals = []
    loss_values = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            labels = y[:, 1:].clone().detach()
            labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            
            with torch.no_grad():
                outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
                loss = outputs[0]
            
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=max_len, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            
            if _%10000==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
            loss_values.append(loss.item())
    ppl_scores = np.exp(np.mean(loss_values))
    print(f'Evaluation Loss:  {np.mean(loss_values)}, perplexity: {ppl_scores}, time: {datetime.now()}')
    return predictions, actuals

sent_tag = pd.read_csv("/content/three_sentenced_data.csv")
sent_tag.head()
X_train, X_test, y_train, y_test = train_test_split(sent_tag["Inputs"], sent_tag["Labels"], test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

train_data = pd.DataFrame(list(zip(X_train, y_train)),columns =['input', 'output'])
train_data.head()
train_data.to_csv('/content/train_data.csv')
test_data = pd.DataFrame(list(zip(X_test, y_test)),columns =['input', 'output'])
test_data.head()
test_data.to_csv('/content/test_data.csv')
val_data = pd.DataFrame(list(zip(X_val, y_val)),columns =['input', 'output'])
val_data.head()
val_data.to_csv('/content/val_data.csv')

def main():
    TRAIN_BATCH_SIZE = 1  # input batch size for training (default: 64)
    VALID_BATCH_SIZE = 1    # input batch size for testing (default: 1000)
    TRAIN_EPOCHS = 5      # number of epochs to train (default: 10)
    VAL_EPOCHS = 1 
    LEARNING_RATE = 1e-4    # learning rate (default: 1e-4)
    SEED = 42               # random seed (default: 42)
    MAX_LEN = 512
    OUTPUT_LEN = 200 

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    np.random.seed(SEED) # numpy random seed
    torch.backends.cudnn.deterministic = True

    random.seed(SEED)
    
    data_dir = '/content/'
    
    # prepare train / valid / test data
    train_df = pd.read_csv(data_dir+'train_data.csv')
    valid_df = pd.read_csv(data_dir+'val_data.csv')
    test_df = pd.read_csv(data_dir+'test_data.csv')

    print("TRAIN Dataset: {}".format(train_df.shape))
    print("VALID Dataset: {}".format(valid_df.shape))
    print("TEST Dataset: {}".format(test_df.shape))
    
    
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = model.to(device)
    
    # Creating the Training / Validation / test dataset for further creation of Dataloader
    training_set = CustomTrainDataset(train_df, tokenizer, MAX_LEN, OUTPUT_LEN)
    val_set = CustomTestDataset(valid_df, tokenizer, MAX_LEN, OUTPUT_LEN)
    test_set = CustomTestDataset(test_df, tokenizer, MAX_LEN, OUTPUT_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }
    
    test_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for training and validation.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)
   
    #model = nn.DataParallel(model)

    # Defining the optimizer that will be used to tune the weights of the network in the training session. 
    optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    model_path = '../../saved_model/Flan_T5_large/'
    min_loss = 1e6
    for epoch in range(TRAIN_EPOCHS):
        loss, perplexity = train(epoch, tokenizer, model, device, training_loader, optimizer)
        print(f'Epoch: {epoch} full results: Loss:  {loss}, perplexity: {perplexity}, time: {datetime.now()}')
        # Save trained model
        if loss < min_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_path + "saved_model")
            min_loss = loss
            # model.save_pretrained('learned_model/T5-S2Q-{}'.format(epoch))
    
    print('Now unmasked data')
    for epoch in range(VAL_EPOCHS):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader, MAX_LEN)
        final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})
        final_df.to_csv('../../results/Flan_T5_large/predictions.csv')
        print('Output Files generated for review')

if __name__ == '__main__':
    main()
    