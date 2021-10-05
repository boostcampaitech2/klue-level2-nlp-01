#%%
import pickle
import torch
from operator import itemgetter
from datasets import load_dataset
import pandas as pd

typed_entity_marker = ['[S:PER]', '[/S:PER]', '[S:ORG]', '[/S:ORG]',
                           '[O:PER]', '[/O:PER]', '[O:ORG]', '[/O:ORG]',
                           '[O:DAT]', '[/O:DAT]', '[O:LOC]', '[/O:LOC]',
                           '[O:POH]', '[/O:POH]', '[O:NOH]', '[/O:NOH]','[S:LOC]','[/S:LOC]']

with open("./dict_label_to_num.pkl", "rb") as f:
    l2n = pickle.load(f)


class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_data(dataset_dir, version=2):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    dataset = load_dataset('csv', data_files=[dataset_dir])['train']
    
    if version == 1 :
        func = data_preprocess1
    elif version == 2 :
        func = data_preprocess2
    else :
        raise ValueError("Version 1=baseline approach , 2=typed-entity-marker approach")
    
    dataset = dataset.map(func, num_proc=4, remove_columns=["id", "subject_entity", "object_entity", "source"])
    return dataset


def data_preprocess1(example):
    """ baseline approach (entity span) """
    example["label"] = l2n[example["label"]]
    example["subject_entity"] = eval(example["subject_entity"])
    example["object_entity"] = eval(example["object_entity"])
    return {"entity_span": example["subject_entity"]["word"] + "[SEP]" + example["object_entity"]["word"]}


def data_preprocess2(example):
    """ typed-entity-marker """
    sentence = example["sentence"]
    example["label"] = l2n[example["label"]]
    example["subject_entity"] = eval(example["subject_entity"])
    example["object_entity"] = eval(example["object_entity"])
    sub_w, sub_s, sub_e, sub_t = example["subject_entity"].values()
    ob_w, ob_s, ob_e, ob_t = example["object_entity"].values()
    tokens = [
      (sub_s, f'[S:{sub_t}]'),
      (sub_e+1, f'[/S:{sub_t}]'),
      (ob_s, f'[O:{ob_t}]'),
      (ob_e+1, f'[/O:{ob_t}]'),
    ]
    tokens.sort(key=lambda x: x[0], reverse=True)
    for start_index, token in tokens:
        sentence = ''.join([sentence[:start_index], token, sentence[start_index:]])
    example["sentence"] = sentence
    return example



def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False,
        ) 
    return tokenized_sentences

'''
================================================================================
'''
with open("./dict_label_to_num.pkl", "rb") as f:
    l2n = pickle.load(f)
    
def preprocessing_dataset2(dataset, label_type):
    label = []
    for i in dataset['label']:
        if not i:
            label.append(100)
        else:
            label.append(label_type[i])
    
    def entity(sent, s1, e1, s2, e2, sub_t, obj_t):
        if s1 < s2:
            return sent[:s1] + f'[S:{sub_t}]' + sent[s1:e1+1] + f'[/S:{sub_t}]' + sent[e1+1:s2] + f'[O:{obj_t}]' + sent[s2:e2+1] + f'[/O:{obj_t}]' + sent[e2+1:]
        else:
            return sent[:s2] + f'[O:{obj_t}]' + sent[s2:e2+1] + f'[/O:{obj_t}]' + sent[e2+1:s1] + f'[S:{sub_t}]' + sent[s1:e1+1] +  f'[/S:{sub_t}]' + sent[e1+1:]

    sub_entity, sub_start, sub_end, sub_type = [], [], [], []
    obj_entity, obj_start, obj_end, obj_type = [] ,[], [], []
    for i in dataset['subject_entity']:
        sub_entity.append(eval(i)['word'])
        sub_start.append(eval(i)['start_idx'])
        sub_end.append(eval(i)['end_idx'])
        sub_type.append(eval(i)['type'])
    for i in dataset['object_entity']:
        obj_entity.append(eval(i)['word'])
        obj_start.append(eval(i)['start_idx'])
        obj_end.append(eval(i)['end_idx'])
        obj_type.append(eval(i)['type'])
    
    sub_frame = pd.DataFrame({'entity_01' : sub_entity, 'entity_01_s' : sub_start, 'entity_01_e': sub_end, 'entity_01_type' : sub_type})
    obj_frame = pd.DataFrame({'entity_02' : sub_entity, 'entity_02_s' : sub_start, 'entity_02_e': sub_end, 'entity_02_type' : obj_type})
    out_dataset = pd.DataFrame(
        {'sentence': dataset['sentence'],'label': label})
    
    out_dataset = pd.concat([out_dataset,sub_frame,obj_frame],axis=1)
    #print(out_dataset)
    
    out_dataset['sentence'] = out_dataset.apply(lambda x: entity(x['sentence'],
                                                                 x['entity_01_s'], x['entity_01_e'],
                                                                 x['entity_02_s'], x['entity_02_e'], x['entity_01_type'],x['entity_02_type']), axis=1)
    return out_dataset


def load_data2(dataset_dir, label_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    with open(label_dir, "rb") as f:
        label_type = pickle.load(f)
        
    pd_dataset = pd.read_csv(dataset_dir)

    dataset = preprocessing_dataset2(pd_dataset, label_type)
    return dataset

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 e1_mask, e2_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask
        
def tokenized_dataset2(dataset, tokenizer, mask_padding_with_zero, max_length):
    if max_length is None: 
        max_length = 150

    features = []
    max_pos = 0

    for idx, sample in dataset.iterrows():
        #print(sample['sentence'])
        tokens = tokenizer.tokenize(sample['sentence'])
        l = len(tokens)
        
        
        e1s = tokens.index('[S:' + sample['entity_01_type'] + ']')
        e1e = tokens.index('[/S:' + sample['entity_01_type'] + ']')
        
        e2s = tokens.index('[O:' + sample['entity_02_type'] + ']')
        e2e = tokens.index('[/O:' + sample['entity_02_type'] + ']')
        
        max_pos = max(max_pos, e1s, e2s)
        #print(max_pos, e1s, e1e, e2s, e2e)
        
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens) #segment_ids
        segment_ids[0] = 1
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)


        e1_mask = [0 for _ in range(len(input_mask))]
        e2_mask = [0 for _ in range(len(input_mask))]
        for i in range(e1s, e1e):
            e1_mask[i] = 1
        for i in range(e2s, e2e):
            e2_mask[i] = 1

        if padding_length < 0:
            input_ids = input_ids[:max_length-1] + tokenizer.convert_tokens_to_ids(["[SEP]"])
            input_mask = input_mask[:max_length-1] + [1 if mask_padding_with_zero else 0]
            segment_ids = segment_ids[:max_length-1] + [0]
            e1_mask = e1_mask[:max_length-1] + [0]
            e2_mask = e2_mask[:max_length-1] + [0]

            
        
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(segment_ids) == max_length


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          e1_mask=e1_mask,
                          e2_mask=e2_mask,
                          segment_ids=segment_ids,
                          label_id=sample['label']))
    return features