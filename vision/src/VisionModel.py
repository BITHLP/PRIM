import os
import sys
sys.path.append(os.path.abspath("../../"))
from torch import nn
import torch
from vision.src.Model import TiMMViTDecoder, TiMMViTEncoder
from vision.src.Model import TransformerDecoder, Embedding, OutputLayer
from vector_quantize_pytorch import VectorQuantize


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.back_encdec = DoubleDecoder(config["back_patch"], config["back_dim"],
                                        config["back_l"], config["back_head"], config["img_h"], config["img_w"])
        self.code_encdec = DoubleQuantDecoder(config["code_patch"], config["code_dim"],
                                        config["code_l"], config["code_head"],
                                        config["codebook_dim"], config["codebook_size"], config["img_h"], config["img_w"])
        self.img_dec = TiMMViTDecoder(config["img_dim"], config["img_l"], config["img_head"], config["img_patch"], (config["img_h"], config["img_w"]))
        self.multitask_encdec = MultiTaskViT(config["mt_d_model"], config["mt_d_ff"], config["mt_n_head"], config["mt_l"], 
                                             config["mt_patch"], config["img_h"], config["img_w"], config["mt_dropout"], config["mt_subwordset"])
        self.src_language_tags = config["src_language_tags"]
        self.tgt_language_tags = config["tgt_language_tags"]

    def forward(self, src_img, tgt_img, src_text, tgt_text):
        # src_img -> rec -> back_img
        # tgt_img -> quant rec -> text_img
        # back_hidden + code_hidden -> tgt_img
        back_dict = self.back_encdec(src_img)
        code_dict = self.code_encdec(tgt_img)
        back_hidden = back_dict["decoder_2_hidden"].detach()
        code_hidden = code_dict["decoder_2_hidden"].detach()
        img = self.img_dec(back_hidden + code_hidden)
        mt_dict = self.multitask_encdec(src_img, src_text, tgt_text)
        return {"back_dict": back_dict, "code_dict": code_dict, "img": img, "mt_dict": mt_dict}

    @torch.no_grad()
    def inference_quant_code(self, img):
        code_dict = self.code_encdec(img)
        return {"code": code_dict["code"]}

    @torch.no_grad()
    def inference_code_hidden(self, code):
        code_dict = self.code_encdec.inference_with_code(code)
        return {"code_2": code_dict["decoder_2_img"], "code_2_hidden": code_dict["decoder_2_hidden"]}
    
    @torch.no_grad()
    def inference_back_hidden(self, img):
        back_dict = self.back_encdec(img)
        return {"back_2": back_dict["decoder_2_img"], "back_2_hidden": back_dict["decoder_2_hidden"]}
    
    @torch.no_grad()
    def inference_output_img(self, back_hidden, code_hidden):
        return self.img_dec(back_hidden + code_hidden)

    @torch.no_grad()
    def inference_tit(self, img, tgt_text):
        return self.multitask_encdec.infer_tit(img, tgt_text)

    @torch.no_grad()
    def greedy_decode_tit(self, img, lang_tag, max_len=64):
        assert lang_tag in self.tgt_language_tags, f"Do not support language {lang_tag}."
        return self.multitask_encdec.greedy_decode_tit(img, lang_tag, max_len)

    @torch.no_grad()
    def greedy_decode_ocr(self, img, lang_tag, max_len=64):
        assert lang_tag in self.src_language_tags, f"Do not support language {lang_tag}."
        return self.multitask_encdec.greedy_decode_ocr(img, lang_tag, max_len)


class DoubleDecoder(nn.Module):
    def __init__(self, patch_size, dim, l, head, img_h, img_w):
        super().__init__()
        self.encoder = TiMMViTEncoder(dim, l, head, patch_size, (img_h, img_w))
        self.decoder_1 = TiMMViTDecoder(dim, l, head, patch_size, (img_h, img_w))
        self.decoder_2 = TiMMViTDecoder(dim, l, head, patch_size, (img_h, img_w))
    
    def forward(self, img):
        enc_hidden = self.encoder(img)
        decoder_1_dict = self.decoder_1.get_hidden(enc_hidden)
        decoder_1_hidden = decoder_1_dict["hidden"]
        decoder_1_img = decoder_1_dict["img"]
        
        decoder_2_dict = self.decoder_2.get_hidden(decoder_1_hidden)
        decoder_2_hidden = decoder_2_dict["hidden"]
        decoder_2_img = decoder_2_dict["img"]

        return {"decoder_1_img": decoder_1_img, "decoder_2_img": decoder_2_img, "decoder_2_hidden": decoder_2_hidden}


class DoubleQuantDecoder(nn.Module):
    def __init__(self, patch_size, dim, l, head, codebook_dim, codebook_size, img_h, img_w):
        super().__init__()
        self.encoder = TiMMViTEncoder(dim, l, head, patch_size, (img_h, img_w))
        self.decoder_1 = TiMMViTDecoder(dim, l, head, patch_size, (img_h, img_w))
        self.decoder_2 = TiMMViTDecoder(dim, l, head, patch_size, (img_h, img_w))
        self.codebook = VectorQuantize(dim=dim, codebook_dim=codebook_dim, codebook_size=codebook_size)
    
    def forward(self, img):
        enc_hidden = self.encoder(img)
        decoder_1_dict = self.decoder_1.get_hidden(enc_hidden)
        decoder_1_hidden = decoder_1_dict["hidden"]
        decoder_1_img = decoder_1_dict["img"]
        quant, code, loss = self.codebook(decoder_1_hidden)
        decoder_2_dict = self.decoder_2.get_hidden(quant)
        decoder_2_hidden = decoder_2_dict["hidden"]
        decoder_2_img = decoder_2_dict["img"]

        return {"decoder_1_img": decoder_1_img, "decoder_2_img": decoder_2_img, "decoder_2_hidden": decoder_2_hidden, "code": code, "vqloss": loss}

    @torch.no_grad()
    def inference_with_code(self, code):
        quant = self.codebook.get_output_from_indices(code)
        decoder_2_dict = self.decoder_2.get_hidden(quant)
        decoder_2_hidden = decoder_2_dict["hidden"]
        decoder_2_img = decoder_2_dict["img"]

        return {"decoder_2_img": decoder_2_img, "decoder_2_hidden": decoder_2_hidden}


class MultiTaskViT(nn.Module):
    def __init__(self, d_model, d_ff, n_head, l, patch, img_h, img_w, dropout, subwordset):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_head = n_head
        self.l = l
        self.patch = patch
        self.dropout = dropout

        self.subwordset = subwordset
        self.padding_idx = self.subwordset["<pad>"]
        self.num_vocab = len(self.subwordset)
        self.img_size = (img_h, img_w)
        
        self.encoder = TiMMViTEncoder(dim=self.d_model, depth=self.l, heads=self.n_head, patch=self.patch, img_size=self.img_size)
        self.ocr_decoder = TransformerDecoder(d_model=self.d_model, d_ff=self.d_ff, n_head=self.n_head, l=self.l, dropout=self.dropout)
        self.embedding = Embedding(d_model=self.d_model, num_vocab=self.num_vocab, padding_idx=self.padding_idx)
        self.ocr_output_layer = OutputLayer(d_model=self.d_model, num_vocab=self.num_vocab)
        self.tit_decoder = TransformerDecoder(d_model=self.d_model, d_ff=self.d_ff, n_head=self.n_head, l=self.l, dropout=self.dropout)
        self.tit_output_layer = OutputLayer(d_model=self.d_model, num_vocab=self.num_vocab)

    def forward(self, img, src_text, tgt_text):
        encoder_hidden = self.encoder(img)
        src_text_embedding = self.embedding(src_text)
        tgt_text_embedding = self.embedding(tgt_text)
        ocr_decoder_hidden = self.ocr_decoder(encoder_hidden, src_text_embedding, y_padding_mask = (src_text == self.padding_idx))
        ocr_output = self.ocr_output_layer(ocr_decoder_hidden)
        tit_decoder_hidden = self.tit_decoder(encoder_hidden, tgt_text_embedding, y_padding_mask = (tgt_text == self.padding_idx))
        tit_output = self.tit_output_layer(tit_decoder_hidden)

        return {"ocr_text": ocr_output, "tit_text": tit_output}

    @torch.no_grad()
    def infer_tit(self, img, tgt_text):
        encoder_hidden = self.encoder(img)
        tgt_text_embedding = self.embedding(tgt_text)
        padding_mask = (tgt_text == self.padding_idx)
        tit_decoder_hidden = self.tit_decoder(encoder_hidden, tgt_text_embedding, y_padding_mask = padding_mask)
        tit_output = self.tit_output_layer(tit_decoder_hidden)
        return {"output_text": tit_output, "hidden_text": tit_decoder_hidden, "padding_mask": padding_mask}

    @torch.no_grad()
    def greedy_decode_ocr(self, img, lang_tag, max_len=64):
        batch_size = img.shape[0]
        device = img.device
        output_text_id = torch.full((batch_size, 1), self.subwordset[lang_tag], dtype=torch.long, device=device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        encoder_hidden = self.encoder(img)
        for _ in range(max_len):
            text_embedding = self.embedding(output_text_id)
            padding_mask = (output_text_id == self.padding_idx)
            decoder_hidden = self.ocr_decoder(encoder_hidden, text_embedding, y_padding_mask=padding_mask)
            output = self.ocr_output_layer(decoder_hidden).argmax(-1) # batch, seq
            next_token = output[:, -1]
            
            is_finished = is_finished | (next_token == self.subwordset["<eos>"])
            next_token[is_finished] = self.padding_idx  # set <pad> for finished sequence
            output_text_id = torch.cat([output_text_id, next_token.unsqueeze(-1)], dim=-1)
            if is_finished.all():
                break
        return {"text": output_text_id.tolist()}

    @torch.no_grad()
    def greedy_decode_tit(self, img, lang_tag, max_len=64):
        batch_size = img.shape[0]
        device = img.device
        output_text_id = torch.full((batch_size, 1), self.subwordset[lang_tag], dtype=torch.long, device=device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        encoder_hidden = self.encoder(img)
        for _ in range(max_len):
            text_embedding = self.embedding(output_text_id)
            padding_mask = (output_text_id == self.padding_idx)
            decoder_hidden = self.tit_decoder(encoder_hidden, text_embedding, y_padding_mask=padding_mask)
            output = self.tit_output_layer(decoder_hidden).argmax(-1) # batch, seq
            next_token = output[:, -1]
            
            is_finished = is_finished | (next_token == self.subwordset["<eos>"])
            next_token[is_finished] = self.padding_idx  # set <pad> for finished sequence
            output_text_id = torch.cat([output_text_id, next_token.unsqueeze(-1)], dim=-1)
            if is_finished.all():
                break
        padding_mask = (output_text_id == self.padding_idx)
        hidden = self.tit_decoder(encoder_hidden, self.embedding(output_text_id), y_padding_mask=padding_mask)
        return {"text": output_text_id.tolist(), "hidden": hidden, "padding_mask": padding_mask}

    def decode_text(self, text_id, lang_tag, spm=None):
        if spm is None:
            idx2char = {v: k for k, v in self.subwordset.items()}
            for each_text in text_id:
                yield "".join([idx2char[i] for i in each_text if (i != self.padding_idx and i != self.subwordset["<eos>"] and i!= self.subwordset[lang_tag])])
        else:
            for each_text in text_id:
                yield spm.decode(each_text)
