# InstructionLLM-QnA
Aims on building a LLM that is task specific and helpful in QnA based tasks

## Table of Contents
- [About](#about)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Approaches to train the data](#approaches-to-train-the-data)
    - [Code](#code-folder)
    - [Dialog Inpainting](#dialog-inpainting)
- [Acknowledgments](#acknowledgments)

## About
This repo contains the code based on research on Large Learning models on QnA based data. In the research, we have used the public dataset from [Taskmaster repo](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-3-2020). The data is preprocessed and later train on [Flan-T5 model](https://huggingface.co/docs/transformers/model_doc/flan-t5).

## Requirement
1. python 3.11 or above
2. pytorch
3. Necessary packages to be installed using the requirements.txt file. using-
```python
pip install -r requirements.txt
```

## How to Run
1. Navigate to code folder and run the data.py file.
```python
python data.py
```
2. Notice that three_sentenced_data.csv file should have been created.
3. Next run the model.py file.
```python
python model.py
```
4. You must note the following files created
    1. saved_model - This file will contain the flan_t5 model
    2. results - This file contains the predictions

## Approaches to train the data
### Code folder
Inside the code folder, the data is trained in following format - 

If the Conversation is C1, C2, C3, C4, C5, C6, C7 ...
| Instance | Training | Labels |
|:--------:|:--------:|:------:|
| First Instance | C1 <mask> C3 | C1 C2 C3 |
| Second Instance | C3 <mask> C5 | C3 C4 C5 |
| Third Instance | C5 <mask> C7 | C5 C6 C7 |

:book: It is to note that only the conversation by the bot in the Taskmaster Dataset is captured.

### Dialog Inpainting
TBA

## Acknowledgments
My sincere gratitude to my professor [Procheta Sen](https://procheta.github.io/sprocheta/) for actively participating in my research. Her support, guidance, and expertise have shaped its success, inspiring me to strive for academic excellence. I am deeply grateful for their invaluable mentorship, fostering my passion and personal growth.