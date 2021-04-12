import torch
from torch import nn


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, dr_rate=None, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, segment_ids, attention_mask):
        _, pooler = self.bert(input_ids=token_ids,
                              token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        return self.classifier(self.dropout(pooler))
