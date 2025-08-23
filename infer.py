from translation.src.TransModel import Translation
from translation.src.utils import load_config
from vision.src.VisionModel import Vision
from FullModel import FullModel
import os
from PIL import Image
from torchvision import utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import sentencepiece as sp
import time
import argparse

class TestDataset(Dataset):
    def __init__(self, img_path, spm, sub_spm):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
        ])
        self.text_sp = spm
        self.subwordset = {}
        self.vocab_size = self.text_sp.get_piece_size()
        for i in range(self.vocab_size):
            self.subwordset[self.text_sp.id_to_piece(i)] = i

        self.sub_text_sp = sub_spm
        self.sub_subwordset = {}
        self.sub_vocab_size = self.sub_text_sp.get_piece_size()
        for i in range(self.sub_vocab_size):
            self.sub_subwordset[self.sub_text_sp.id_to_piece(i)] = i

        img_list = sorted(os.listdir(img_path), key=lambda x: int(x.split('.')[0]))
        self.img_list = [os.path.join(img_path, img) for img in img_list]
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = self.transform(img)
        return img



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True, default=1)
    parser.add_argument("--input_img_dir", type=str, required=True)
    parser.add_argument("--output_img_dir", type=str, required=True)
    parser.add_argument("--lang_tag", type=str, required=True)
    args = parser.parse_args()

    data_config, train_config, trans_config = load_config(args.config)

    spm = sp.SentencePieceProcessor(data_config["spm_path"])
    sub_spm = sp.SentencePieceProcessor(data_config["subspm_path"])

    ds = TestDataset(args.input_img_dir, spm, sub_spm)
    dl = DataLoader(ds, args.batch_size, shuffle=False)
    tgtimg_dir = args.output_img_dir
    lang_tag = args.lang_tag

    if not os.path.exists(tgtimg_dir):
        os.makedirs(tgtimg_dir)
    
    vision_config_path = train_config["vision_config"]
    vision_ckpt_path = train_config["vision_ckpt"]

    _, _, vision_config = load_config(vision_config_path)
    vision_config["mt_subwordset"] = ds.subwordset
    vision_config["src_language_tags"] = data_config["src_language_tags"]
    vision_config["tgt_language_tags"] = data_config["tgt_language_tags"]

    trans_config["vision_d_model"] = vision_config["mt_d_model"]
    trans_config["vision_d_model"] = vision_config["mt_d_model"]
    trans_config["sub_subwordset"] = ds.sub_subwordset
    trans_config["max_seq_len"] = data_config["max_subtext_len"]
    
    trans_ckpt = torch.load(args.checkpoint, map_location="cpu")
    trans_model = Translation(trans_config)
    trans_model.load_state_dict(trans_ckpt["model_state"])
    trans_model.eval().cuda()

    vision_ckpt = torch.load(vision_ckpt_path, map_location="cpu")
    vision_model = Vision(vision_config)
    vision_model.load_state_dict(vision_ckpt["model_state"])
    vision_model.eval().cuda()

    full_model = FullModel(vision_model, trans_model)
    full_model.eval().cuda()

    code_len = int((vision_config["img_h"] * vision_config["img_w"]) / (vision_config["img_patch"] * vision_config["img_patch"]))
    start = time.time()
    idx = 0
    for img in dl:
        img = img.cuda()
        tgt_imgs = full_model.inference(img, lang_tag, code_len)
        for tgt_img in tgt_imgs:
            vutils.save_image(tgt_img*0.5+0.5, os.path.join(tgtimg_dir, f"{idx}.jpg"))
            idx += 1
    end = time.time()
    print(f"Decoding time: {end-start}")
