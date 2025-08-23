import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from PIL import Image
import sentencepiece as sp
import lmdb
import pickle
import numpy as np


class LMDBTransDataset(Dataset):
    def __init__(self, lmdb_path_list, spm_path, sub_spm_path, code_file_path_list, code_bos, code_eos, src_language_tags,tgt_language_tags, 
                 max_text_len=24, max_subtext_len=48, subtext_group_size=1):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
        ])
        self.text_sp = sp.SentencePieceProcessor(spm_path)
        self.subwordset = {}
        self.vocab_size = self.text_sp.get_piece_size()
        for i in range(self.vocab_size):
            self.subwordset[self.text_sp.id_to_piece(i)] = i

        self.sub_text_sp = sp.SentencePieceProcessor(sub_spm_path)
        self.sub_subwordset = {}
        self.sub_vocab_size = self.sub_text_sp.get_piece_size()
        for i in range(self.sub_vocab_size):
            self.sub_subwordset[self.sub_text_sp.id_to_piece(i)] = i

        self.orig_max_text_len = max_text_len
        self.orig_max_subtext_len = max_subtext_len

        self.max_text_len = max_text_len + 1
        self.max_subtext_len = max_subtext_len + subtext_group_size
        self.subtext_group_size = subtext_group_size

        self.src_language_tags_id = []
        self.tgt_language_tags_id = []

        for src_tag in src_language_tags:
            assert src_tag in self.subwordset.keys()
            self.src_language_tags_id.append(self.subwordset[src_tag])
        
        for tgt_tag in tgt_language_tags:
            assert tgt_tag in self.subwordset.keys()
            self.tgt_language_tags_id.append(self.subwordset[tgt_tag])
        
        self.envs = [lmdb.open(path, readonly=True, lock=False) for path in lmdb_path_list]

        code_list = []
        for path in code_file_path_list:
            f = open(path, "r")
            for l in f:
                code_list.append([int(c) for c in l.strip().split()])

        self.env_keys = []

        for env_id in range(len(self.envs)):
            env = self.envs[env_id]
            with env.begin(write=False) as txn:
                keys = list(txn.cursor().iternext(values=False))
                for key in keys:
                    self.env_keys.append((env_id, key))

        bos_code_list = [] # including bos
        eos_code_list = [] # including eos
        for code in code_list:
            bos_code_list.append([code_bos] + code)
            eos_code_list.append(code + [code_eos])
        
        self.code_list = torch.tensor(bos_code_list)
        self.label_code_list = torch.tensor(eos_code_list)

    def __len__(self):
        return len(self.env_keys)

    def __getitem__(self, idx):
        code = self.code_list[idx]
        label_code = self.label_code_list[idx]
        env_id = self.env_keys[idx][0]
        env = self.envs[env_id]
        env_key = self.env_keys[idx][1]
        with env.begin(write=False) as txn:
            value = txn.get(env_key)
            data = pickle.loads(value)
            src_img = Image.frombytes(data["src_img_mode"], data["src_img_size"], data["src_img"])
            text = data["tgt_text"].decode()
            idx_tgt_text = self.text_sp.EncodeAsIds(text.strip())
            idx_tgt_subtext = self.sub_text_sp.EncodeAsIds(text.strip())
            assert len(idx_tgt_text) <= self.orig_max_text_len , f"length of tgt text is {len(idx_tgt_text)} > {self.orig_max_text_len}"
            assert len(idx_tgt_subtext) <= self.orig_max_subtext_len, f"length of sub text is {len(idx_tgt_subtext)} > {self.orig_max_subtext_len}"

            bos_text = [self.tgt_language_tags_id[env_id]] + idx_tgt_text
            text_eos = idx_tgt_text + [self.subwordset["<eos>"]]
            tgt_text = bos_text + [self.subwordset["<pad>"]] * (self.max_text_len - len(bos_text))
            label_tgt_text = text_eos + [self.subwordset["<pad>"]] * (self.max_text_len - len(text_eos))

            bos_sub_text = [self.sub_subwordset["<bos>"]] * self.subtext_group_size + idx_tgt_subtext
            sub_text_eos = idx_tgt_subtext + [self.sub_subwordset["<eos>"]] * self.subtext_group_size
            sub_text = bos_sub_text + [self.sub_subwordset["<pad>"]] * (self.max_subtext_len - len(bos_sub_text))
            label_sub_text = sub_text_eos + [self.sub_subwordset["<pad>"]] * (self.max_subtext_len - len(sub_text_eos))

        return self.transform(src_img), torch.tensor(tgt_text), torch.tensor(label_tgt_text), code, label_code, torch.tensor(sub_text), torch.tensor(label_sub_text)
    