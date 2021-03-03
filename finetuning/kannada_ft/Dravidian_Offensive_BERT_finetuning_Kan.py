import pandas as pd
import numpy as np
import torch

# Reading CSV from link
def read_csv_from_link(url):
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    df = pd.read_csv(path,delimiter="\t",error_bad_lines=False, header=None)
    return df

# Loading All Data
kannada_train = read_csv_from_link('https://drive.google.com/file/d/1BFYF05rx-DK9Eb5hgoIgd6EcB8zOI-zu/view?usp=sharing')
kannada_dev = read_csv_from_link('https://drive.google.com/file/d/1V077dMQvscqpUmcWTcFHqRa_vTy-bQ4H/view?usp=sharing')
# Mal Preprocess
kannada_train = kannada_train.iloc[:, 0:2]
kannada_train = kannada_train.rename(columns={0: "text", 1: "label"})
# Stats
kannada_train['label'] = pd.Categorical(kannada_train.label)

# Mal Preprocess
kannada_dev = kannada_dev.iloc[:, 0:2]
kannada_dev = kannada_dev.rename(columns={0: "text", 1: "label"})
# Stats
kannada_dev['label'] = pd.Categorical(kannada_dev.label)

# Change Device - CPU/GPU-0/GPU-1
torch.cuda.set_device(0)
device = 'cuda'
device = device if torch.cuda.is_available() else 'cpu'

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset

# Dataset
class kannada_Offensive_Dataset(Dataset):
    def __init__(self, encodings, labels, bpe = False):
        self.encodings = encodings
        self.labels = labels
        self.is_bpe_tokenized = bpe

    def __getitem__(self, idx):
        if not self.is_bpe_tokenized:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        else:
            item = {
                'input_ids': torch.LongTensor(self.encodings[idx].ids),
                'attention_mask': torch.LongTensor(self.encodings[idx].attention_mask)
            }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

models = ['MURIL', 'XLMR', 'mbertlarge', 'XLMR_custom', 'mbertbase', 'indic', 'XLMR_base']
wbools = [True, False]

import itertools
for model, wbool in list(itertools.product(models, wbools)):
    loss_weighted = wbool
    
    if model == 'MURIL':
        # Using Huggingface MURIL version
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("simran-kh/muril-cased-temp")
        model = AutoModelForSequenceClassification.from_pretrained("simran-kh/muril-cased-temp", num_labels=6)
        model_name = 'MURIL_cased_temp_kannada'
    if model == 'XLMR':
        # Using XLM-Roberta-Base pretrained model
        from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large', num_labels=6)
        model_name = 'XLMroberta_large_kannada'
    if model == 'XLMR_base':
        # Using XLM-Roberta-Base pretrained model
        from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=6)
        model_name = 'XLMroberta_base_kannada'
    if model == 'mbertlarge':
        # Using Multilingual Bert, bert-large-multilingual-cased pretrained
        from transformers import BertTokenizer, BertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        model = BertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=6)
        model_name = 'Distilbert_m_base_cased_kannada'
    if model == 'XLMR_custom':
        # Using XLMRoberta finetuning Custom Pretrained model, Vocab same => Tokenizer base
        from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        model = XLMRobertaForSequenceClassification.from_pretrained('../../pretrained_models/XLMroberta_best_pretrained_kannada/', num_labels=6)
        model_name = 'XLMroberta_custom_pretrained_kannada'
    if model == 'mbertbase':
        # Using Multilingual Bert, bert-base-multilingual-cased pretrained
        from transformers import BertTokenizer, BertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=6)
        model_name = 'Mbert_base_cased_kannada'
    if model == 'indic':        
        # Using Indic Bert
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
        model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels = 6)
        model_name = 'Indic_bert_kannada'
    
    if loss_weighted:
        model_name = model_name + '_weighted'
    print("Model: {}".format(model_name))

    # Optimiser
    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-5)

    label_mapping = {
        'Not_offensive': 0, 
        'not-Kannada': 1,
        'Offensive_Targeted_Insult_Other': 2,
        'Offensive_Targeted_Insult_Group': 3, 
        'Offensive_Untargetede': 4, 
        'Offensive_Targeted_Insult_Individual': 5
    }

    # Collecting Text and Labels
    train_batch_sentences = list(kannada_train['text'])
    train_batch_labels =  [label_mapping[x] for x in kannada_train['label']]
    dev_batch_sentences = list(kannada_dev['text'])
    dev_batch_labels =  [label_mapping[x] for x in kannada_dev['label']]

    # Convert to Tensor
    if 'parameters' in tokenizer.__dict__.keys() and tokenizer.__dict__['_parameters']['model'] == 'ByteLevelBPE':
        train_encodings = tokenizer.encode_batch(train_batch_sentences)
        dev_encodings = tokenizer.encode_batch(dev_batch_sentences)
    else:
        train_encodings = tokenizer(train_batch_sentences, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        dev_encodings = tokenizer(dev_batch_sentences, padding='max_length', truncation=True, max_length=64, return_tensors="pt")

    train_labels = torch.tensor(train_batch_labels)
    dev_labels = torch.tensor(dev_batch_labels)

    # Defining Datasets
    train_dataset = kannada_Offensive_Dataset(train_encodings, train_labels, bpe = False)
    dev_dataset = kannada_Offensive_Dataset(dev_encodings, dev_labels, bpe = False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    best_val_f1 = 0
    count = 0
        
    # Alternate Loss Fn
    # Weighted Manual Loss Function
    from sklearn.utils import class_weight
    import torch.nn as nn
    weights = class_weight.compute_class_weight('balanced',np.unique(train_batch_labels),train_batch_labels)
    weights = np.exp(weights)/np.sum(np.exp(weights))
    class_weights = torch.FloatTensor(weights).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

    for epoch in range(100):
        train_preds = []
        train_labels = []
        total_train_loss = 0
        model.train()
        print("==========================================================")
        print("Epoch {}".format(epoch))
        print("Train")
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            if loss_weighted:
                loss = loss_function(outputs[1], labels)
            else:
                loss = outputs[0]
            loss.backward()
            optimizer.step()

            for logits in outputs[1].detach().cpu().numpy():
                train_preds.append(np.argmax(logits))
            for logits in labels.cpu().numpy():
                train_labels.append(logits)
            total_train_loss += loss.item()/len(train_loader)

        print("Dev")
        dev_preds = []
        model.eval()
        total_val_loss = 0
        with torch.set_grad_enabled(False):
            for batch in tqdm(dev_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                if loss_weighted:
                    loss = loss_function(outputs[1], labels)
                else:
                    loss = outputs[0]
                total_val_loss += loss.item()/len(dev_loader)

                for logits in outputs[1].cpu().numpy():
                    dev_preds.append(np.argmax(logits))

        y_true = dev_batch_labels
        y_pred = dev_preds
        target_names = label_mapping.keys()
        train_report = classification_report(train_labels, train_preds, target_names=target_names)
        report = classification_report(y_true, y_pred, target_names=target_names)
        val_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Save Best Model
        if val_f1 > best_val_f1:
            PATH = '../../finetuned_models/' + model_name + '.pth'
            torch.save(model.state_dict(), PATH)
            model.save_pretrained(os.path.join('../../finetuned_berts/', model_name))
            best_val_f1 = val_f1
            count = 0
        else:
            count += 1

        print(train_report)
        print(report)
        print("Epoch {}, Train Loss = {}, Val Loss = {}, Val F1 = {}, Best Val f1 = {}, stagnant = {}".format(epoch, total_train_loss, total_val_loss, val_f1, best_val_f1, count))
        if count == 5:
            print("No increase for 5 epochs, Stopping ...")
            break
    
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()