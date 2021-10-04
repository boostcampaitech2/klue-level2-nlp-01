#%%
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

class SequenceClassificationModel(nn.Module):
    def __init__(self, MODEL_NAME='klue/bert-base'):
        super().__init__()

        # setting model hyperparameter
        model_config =  AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = 30
            
        self.model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        
        print(self.model.config)
    
    def forward(self, inputs):
        
        input_ids = inputs["input_ids"] # 배치, 문장
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels
        )
        
        # outputs : SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,  [batch, num_labels]
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        return outputs


class TokenClassificationModel(nn.Module):

    typed_entity_marker = ['[S:PER]', '[/S:PER]', '[S:ORG]', '[/S:ORG]',
                           '[O:PER]', '[/O:PER]', '[O:ORG]', '[/O:ORG]',
                           '[O:DAT]', '[/O:DAT]', '[O:LOC]', '[/O:LOC]',
                           '[O:POH]', '[/O:POH]', '[O:NOH]', '[/O:NOH]',]
    
    def __init__(self, MODEL_NAME='klue/bert-base'):
        super().__init__()
            
        self.model =  AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
        self.model.resize_token_embeddings(self.model.config.vocab_size + len(self.typed_entity_marker))
        
        self.num_labels = 30
        self.hidden_size = self.model.config.hidden_size

        self.token_classifier = nn.Sequentual(
            nn.linear(self.hidden_size, self.num_labels * 2),
            nn.linear(self.num_labels * 2, self.num_labels)
        )
        # print(self.model.config)
    
    def forward(self, inputs):
        
        input_ids = inputs["input_ids"] # 배치, 문장
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        batch_size = input_ids.size(0)
        entity_token_mask = (input_ids >= 32000)  # [batch, seq_len]
        
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True,
        )
        # outputs : TokenClassifierOutput(
        #     loss=loss,
        #     logits=logits, [batch, seq_len, num_labels]
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        token_logits = outputs.hidden_states
        entity_token_logits = token_logits[entity_token_mask].view(batch_size, -1, )  # batch * 4, num_labels
        
        entity_token_logits = entity_token_logits[:, ::2, :].contiguous().view(batch_size, -1) # batch, num_labels * 2


        print(entity_token_logits.shape)
        
        return outputs
        
        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )



# %%

""" 모델 출력 확인용 """

# from transformers import AutoTokenizer
# from load_data import *
# from utils import *

# MODEL_NAME = "klue/bert-base"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": TokenClassificationModel.typed_entity_marker})

# # load dataset
# train_dataset = load_data("../dataset/train/train.csv", data_preprocess2)
# dev_dataset = load_data("../dataset/train/dev.csv", data_preprocess2) # validation용 데이터는 따로 만드셔야 합니다.

# train_label = train_dataset['label']
# dev_label = dev_dataset['label']

# # tokenizing dataset
# def tokenizing(datasets):
#     return tokenizer(
#         # datasets["entity_span"],
#         datasets["sentence"],
#         return_tensors="pt", 
#         truncation=True, 
#         padding="max_length", 
#         max_length=256, 
#         add_special_tokens=True, 
#         return_token_type_ids=True)

# tokenized_train = tokenizing(train_dataset)
# tokenized_dev = tokenizing(dev_dataset)

# # make dataset for pytorch.
# RE_train_dataset = RE_Dataset(tokenized_train, train_label)
# RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

# #%%

# # model = SequenceClassificationModel(MODEL_NAME)
# model = TokenClassificationModel(MODEL_NAME)

# model.forward(RE_train_dataset[:2])