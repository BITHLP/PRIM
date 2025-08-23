import os
import sys
sys.path.append(os.path.abspath("../../"))
from torch import nn
import torch
from translation.src.Model import TransformerEncoder, TransformerDecoder, Embedding, OutputLayer, GroupedTransformerDecoder

class Translation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.code_bos = config["code_bos"]
        self.code_eos = config["code_eos"]
        self.sub_subwordset = config["sub_subwordset"]
        self.sub_text_bos = self.sub_subwordset["<bos>"]
        self.sub_text_eos = self.sub_subwordset["<eos>"]
        self.sub_text_padding = self.sub_subwordset["<pad>"]

        self.group_size = config["group_size"]
        self.code_embedding = Embedding(config["code_num"], config["d_model"], padding_idx=None)
        self.code_decoder = TransformerDecoder(config["d_model"], config["d_ff"], config["n_head"], config["l"], config["dropout"])
        self.output_layer = OutputLayer(config["d_model"], config["code_num"])

        assert config["vision_d_model"] == config["d_model"]
        self.sub_text_embedding = Embedding(len(self.sub_subwordset), config["vision_d_model"], padding_idx=self.sub_subwordset["<pad>"])
        self.sub_text_decoder = GroupedTransformerDecoder(config["vision_d_model"], config["sub_subword_d_ff"], config["n_head"], config["sub_subword_l"], config["group_size"], config["dropout"], config["max_seq_len"])

        self.sub_text_output = OutputLayer(config["vision_d_model"], len(self.sub_subwordset))

    def forward(self, input_hidden, code, sub_text, x_padding_mask=None):
        sub_text_embed = self.sub_text_embedding(sub_text)
        sub_text_padding_mask = (sub_text == self.sub_text_padding)
        sub_text_hidden = self.sub_text_decoder(input_hidden, sub_text_embed, x_padding_mask=x_padding_mask, y_padding_mask=sub_text_padding_mask)
        sub_text_output = self.sub_text_output(sub_text_hidden)

        code_embed = self.code_embedding(code)
        code_hidden = self.code_decoder(sub_text_hidden, code_embed, x_padding_mask=sub_text_padding_mask)
        code_output = self.output_layer(code_hidden)
        return {"code": code_output, "sub_text": sub_text_output}

    @torch.no_grad()
    def sar_decode_sub_text(self, input_hidden, max_len=64, x_padding_mask=None):
        batch_size = input_hidden.size(0)
        device = input_hidden.device

        tgt_sub_text = torch.full((batch_size, self.group_size), self.sub_text_bos, dtype=torch.long, device=device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(0, max_len, self.group_size):
            sub_text_embed = self.sub_text_embedding(tgt_sub_text)
            sub_text_padding_mask = (tgt_sub_text == self.sub_text_padding)
            sub_text_hidden = self.sub_text_decoder(input_hidden, sub_text_embed, x_padding_mask=x_padding_mask, y_padding_mask=sub_text_padding_mask)

            output = self.sub_text_output(sub_text_hidden).argmax(-1) # batch_size, seq_len
            next_group_tokens = output[:, -self.group_size:]
            next_group_tokens[is_finished] = self.sub_text_padding

            next_group_is_eos = ((next_group_tokens == self.sub_text_eos).cumsum(-1) - (next_group_tokens == self.sub_text_eos).int()).bool()
            next_group_tokens[next_group_is_eos] = self.sub_text_padding

            is_finished = is_finished | (next_group_tokens == self.sub_text_eos).any(-1)

            tgt_sub_text = torch.cat([tgt_sub_text, next_group_tokens], dim=-1)
            if is_finished.all():
                break
        
        tgt_sub_text_list = tgt_sub_text[:, 1:].tolist()
        tgt_sub_text_hidden = self.sub_text_decoder(input_hidden, self.sub_text_embedding(tgt_sub_text), x_padding_mask)
        tgt_sub_text_paddimg_mask = (tgt_sub_text == self.sub_text_padding)
        return {"tgt_sub_text": tgt_sub_text_list, "tgt_sub_text_hidden": tgt_sub_text_hidden, "tgt_sub_text_padding_mask": tgt_sub_text_paddimg_mask}

    @torch.no_grad()
    def greedy_decode(self, input_hidden, max_len=64, x_padding_mask=None):
        batch_size = input_hidden.size(0)
        device = input_hidden.device
        sub_text_dict = self.sar_decode_sub_text(input_hidden, max_len, x_padding_mask)
        sub_text_hidden = sub_text_dict["tgt_sub_text_hidden"]
        sub_text_padding_mask = sub_text_dict["tgt_sub_text_padding_mask"]
        tgt_code = torch.full((batch_size, 1), self.code_bos, dtype=torch.long, device=device)

        for _ in range(max_len):
            code_embed = self.code_embedding(tgt_code)
            code_hidden = self.code_decoder(sub_text_hidden, code_embed, sub_text_padding_mask)
            output = self.output_layer(code_hidden).argmax(-1) # batch, vocab
            tgt_code = torch.cat([tgt_code, output[:, -1:]], dim=-1)
        
        return tgt_code[:, 1:]
