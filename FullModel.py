from translation.src.TransModel import Translation
from vision.src.VisionModel import Vision
import torch.nn as nn
import torch


class FullModel(nn.Module):
    def __init__(self, vision_model: Vision, translation_model: Translation):
        super(FullModel, self).__init__()
        self.vision_model = vision_model
        self.translation_model = translation_model

    @torch.no_grad()
    def inference(self, src_img, lang_tag, code_len):
        back_hidden = self.vision_model.inference_back_hidden(src_img)["back_2_hidden"]
        tit_dict = self.vision_model.greedy_decode_tit(src_img, lang_tag)
        code = self.translation_model.greedy_decode(tit_dict["hidden"], max_len=code_len, x_padding_mask=tit_dict["padding_mask"])
        code_hidden = self.vision_model.inference_code_hidden(code)["code_2_hidden"]
        tgt_img = self.vision_model.inference_output_img(back_hidden, code_hidden)

        return tgt_img
