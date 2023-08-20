# This code trains the data on OpenAI's GPT-2
import pandas as pd
import numpy as np
import re
import os

import transformers
from transformers import TextDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

class GPT2Train:
    def preprocess_data(self, path):
        # This function is responsible for creating text file from the json files
        speaker = []
        conv = []
        convID = []
        lst = []
        s = ""
        json_file_names = [filename for filename in os.listdir(self.path) if filename.endswith('.json')]
        with open('data.txt', 'w') as f:
            for json_file_name in json_file_names:
                with open(os.path.join(self.path, json_file_name)) as json_file:
                    data = json.load(json_file)
                    print("Number of conversations: ",len(data))
                    for i in range(len(data)):
                        for j in range(0,len(data[i]['utterances']),2):
                            if j<(len(data[i]['utterances'])-2):
                                s = "User: "+data[i]['utterances'][j]['text'] + " Assistant: " + data[i]['utterances'][j+1]['text']
                                f.write('\n'.join(s))
                                s = ""
                            else:
                                if (len(data[i]['utterances'])-1) % 2 != 0:
                                    break
                                else:
                                    s = "User: "+data[i]['utterances'][j]['text'] + " Assistant: " + data[i]['utterances'][j+1]['text']
                                    f.write('\n'.join(s))
                                    s= ""
                                    break                            
        f.close()

    def load_dataset(file_path, tokenizer, block_size = 128):
        dataset = TextDataset(
            tokenizer = tokenizer,
            file_path = file_path,
            block_size = block_size,
        )
        return dataset

    def train(train_file_path,model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        train_dataset = GPT2Train.load_dataset(train_file_path, tokenizer)

        tokenizer.save_pretrained(output_dir)

        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.save_pretrained(output_dir)

        training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=overwrite_output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                num_train_epochs=num_train_epochs,
            )

        trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
        )
            
        trainer.train()
        trainer.save_model()
    
    def inference_data(model_path, tokenizer_path, sequence, max_length):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        model = load_model(model_path)
        tokenizer = load_tokenizer(model_path)
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        final_outputs = model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )
        print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

## Running the gpt2 model
path = "/content/Taskmaster/TM-3-2020/data"
load_gpt2 = GPT2Train()
load_gpt2.preprocess_data(path)
train_file_path ="/content/data.txt"
model_name = "gpt-2"
out_dir = "/content/model"
overwrite_output_dir = False
per_device_train_batch_size = 8
num_train_epochs = 50.0
save_steps = 50000
load_gpt2.train(
    train_file_path=train_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)

sequence = "User: That’s awesome and I’ll love to take in the 9:10pm showing. Assistant: "
max_len = 50
load_gpt2.inference_data(model1_path, sequence, max_len)
#Most likely answer: Great. And how many tickets? 