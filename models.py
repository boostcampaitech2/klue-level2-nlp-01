#%%
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModel
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput


class SequenceClassificationModel(nn.Module):
    def __init__(self, MODEL_NAME='klue/bert-base'):
        super().__init__()

        # setting model hyperparameter
        model_config =  AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = 30
            
        self.model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        
        print(self.model.config)
    
    def forward(  # Trainer API 가 받는 입력 형태대로 받아서 넘겨주어야 함.
        self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
        inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        
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

    # train.py 에서 이용을 위해 클래스 변수로 선언
    typed_entity_marker = ['[S:PER]', '[/S:PER]', '[S:ORG]', '[/S:ORG]',
                           '[O:PER]', '[/O:PER]', '[O:ORG]', '[/O:ORG]',
                           '[O:DAT]', '[/O:DAT]', '[O:LOC]', '[/O:LOC]',
                           '[O:POH]', '[/O:POH]', '[O:NOH]', '[/O:NOH]',]
    
    def __init__(self, MODEL_NAME='klue/bert-base'):
        super().__init__()
            
        self.model =  AutoModel.from_pretrained(MODEL_NAME)
        self.model.resize_token_embeddings(self.model.config.vocab_size + len(self.typed_entity_marker))
        
        self.num_labels = 30
        self.hidden_size = self.model.config.hidden_size

        # entity token classifier => 참조한 논문에서 이용한 형태와 동일.
        self.token_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Linear(self.hidden_size, self.num_labels)
        )

        print(self.model.config)
    
    def forward(  # Trainer API 가 받는 입력 형태대로 받아서 넘겨주어야 함.
        self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
        inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None,
    ):

        batch_size = input_ids.size(0)
        entity_token_mask = (input_ids >= 32000)  # [batch, seq_len] 엔티티 토큰만을 골라내기 위한 Mask

        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        )
        # pre-trained 모델 출력 형태
        # outputs : TokenClassifierOutput(
        #     loss=loss,       =>   labels 인자를 넣어주지 않기 때문에 loss=None이 반환됨.
        #     logits=logits,   =>   [batch, seq_len, num_labels]
        #     hidden_states=outputs.hidden_states,   =>  Tuple[embedding_output, each_layer_hidden_states]  tuple_length=13
        #     attentions=outputs.attentions,    =>  None
        # 
        token_logits = outputs.last_hidden_state  # 출력형태 => Tuple[embedding_output, each_layer_hidden_states], tuple length=13
        entity_token_logits = token_logits[entity_token_mask].view(batch_size, -1, self.hidden_size)  # batch, 4, hidden_size
        
        entity_token_logits = entity_token_logits[:, ::2, :].contiguous().view(batch_size, -1) # batch, hidden_size * 2
        
        logits = self.token_classifier(entity_token_logits)  # batch, num_labels
        loss = F.cross_entropy(logits, labels, ignore_index=0)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )



# %%

# """ 모델 출력 확인용 """

# from transformers import AutoTokenizer
# from load_data import *
# from utils import *

# MODEL_NAME = "klue/bert-base"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": TokenClassificationModel.typed_entity_marker})

# # load dataset
# train_dataset = load_data("../dataset/train/train.csv", version=2)
# dev_dataset = load_data("../dataset/train/dev.csv", version=2) # validation용 데이터는 따로 만드셔야 합니다.

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
# from transformers import BertForTokenClassification

# # model = SequenceClassificationModel(MODEL_NAME)
# model = TokenClassificationModel(MODEL_NAME)

# outputs = model.forward(**RE_train_dataset[3:8])