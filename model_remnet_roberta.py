import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel, BertPreTrainedModel


BertLayerNorm = torch.nn.LayerNorm


class REMNet(BertPreTrainedModel):
    def __init__(self,
                 config,
                 num_choices: int = 3,
                 recursive_step: int = 2,
                 erasure_k: int = 50):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.num_choices = num_choices

        
        self.recursive = recursive_step
        self.erasure_k = erasure_k

        self.memory_in = nn.MultiheadAttention(embed_dim=config.hidden_size,
                                               num_heads=8,
                                               dropout=0.1)

        self.mem_linear = nn.Linear(config.hidden_size * self.recursive, config.hidden_size)

        self.merge_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.single_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, self.num_choices)
        self.classifier_2 = nn.Linear(config.hidden_size, 1)


    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def memory(self, ms, q, key_padding_mask):
        q = q.unsqueeze(0)
        ms = ms.permute(1, 0, 2)
        att_in, att_in_weights = self.memory_in(query = q,
                                    key = ms,
                                    value = ms,
                                    key_padding_mask = key_padding_mask)

        output = q + att_in
        output = output.squeeze(0)    

        weights = att_in_weights.squeeze(1)    
        lowestk, id = weights.topk(k=self.erasure_k + 1, dim=1, largest=False)    
        lowestk = lowestk[:, -1]    
        lowestk = lowestk.unsqueeze(1).repeat(1, weights.size(1))    
        new_padding_mask = torch.lt(weights, lowestk)    
        new_padding_mask += key_padding_mask    
        new_padding_mask = new_padding_mask.bool()    

        return output, new_padding_mask

    def forward(self, input_ids, token_type_ids, attention_mask, labels, evid_feats, question_types):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        sequence_output, pooled_output = self.roberta(input_ids=flat_input_ids,
                                                      attention_mask=flat_attention_mask)


        if isinstance(evid_feats, list):
            ante_feats, cons_feats = evid_feats

            mem_1 = []
            padding_1 = torch.zeros(size=(ante_feats.size(0), ante_feats.size(1)),    
                                    dtype=torch.bool,
                                    device=ante_feats.device)
            memory_output_1, padding_1 = self.memory(ms=ante_feats, q=pooled_output,
                                                     key_padding_mask=padding_1)    
            mem_1.append(memory_output_1)

            if self.recursive > 1:
                for i in range(self.recursive - 1):
                    memory_output_1, padding_1 = self.memory(ms=ante_feats, q=memory_output_1,
                                                             key_padding_mask=padding_1)
                    mem_1.append(memory_output_1)

            mem_1 = torch.cat(mem_1, dim=-1)    
            mem_1 = self.mem_linear(mem_1)    

            mem_2 = []
            padding_2 = torch.zeros(size=(cons_feats.size(0), cons_feats.size(1)),
                                    dtype=torch.bool,
                                    device=cons_feats.device)
            memory_output_2, padding_2 = self.memory(ms=cons_feats, q=pooled_output,
                                                     key_padding_mask=padding_2)    
            mem_2.append(memory_output_2)

            if self.recursive > 1:
                for i in range(self.recursive - 1):
                    memory_output_2, padding_2 = self.memory(ms=cons_feats, q=memory_output_2,
                                                             key_padding_mask=padding_2)
                    mem_2.append(memory_output_2)

            mem_2 = torch.cat(mem_2, dim=-1)    
            mem_2 = self.mem_linear(mem_2)    

            memory_output = torch.cat([mem_1, mem_2], dim=-1)    
            memory_output = self.merge_linear(memory_output)    

            logits = self.classifier(memory_output)

        else:
            bsz = evid_feats.size(0)

            stacked_evid_feats = torch.stack([evid_feats.unsqueeze(1)] * self.num_choices,
                                             dim=1)    
            flat_evid_feats = stacked_evid_feats.view(-1, stacked_evid_feats.size(-2), stacked_evid_feats.size(
                -1))    

            mem_1 = []
            padding_1 = torch.zeros(size=(flat_evid_feats.size(0), flat_evid_feats.size(1)),    
                                    dtype=torch.bool,
                                    device=flat_evid_feats.device)
            memory_output_1, padding_1 = self.memory(ms=flat_evid_feats, q=pooled_output,
                                                     key_padding_mask=padding_1)    
            mem_1.append(memory_output_1)

            if self.recursive > 1:
                for i in range(self.recursive - 1):
                    memory_output_1, padding_1 = self.memory(ms=flat_evid_feats, q=memory_output_1,
                                                             key_padding_mask=padding_1)
                    mem_1.append(memory_output_1)

            mem_1 = torch.cat(mem_1, dim=-1)    
            mem_1 = self.mem_linear(mem_1)    

            memory_output = self.single_linear(mem_1)    

            logits = self.classifier_2(memory_output)    
            logits = logits.squeeze(-1)
            logits = logits.view(bsz, -1)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return loss, logits
