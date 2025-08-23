import torch
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms
from PIL import Image
import sentencepiece as sp
import lmdb
import pickle


class LMDBVisionDataset(Dataset):
    def __init__(self, lmdb_path_list, spm_path, src_language_tags, tgt_language_tags, max_text_len=24):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
        ])
        self.text_sp = sp.SentencePieceProcessor(spm_path)
        self.subwordset = {}
        self.vocab_size = self.text_sp.get_piece_size()
        for i in range(self.vocab_size):
            self.subwordset[self.text_sp.id_to_piece(i)] = i

        self.orig_max_text_len = max_text_len
        self.max_text_len = max_text_len + 1

        self.src_language_tags_id = []
        self.tgt_language_tags_id = []

        for src_tag in src_language_tags:
            assert src_tag in self.subwordset.keys()
            self.src_language_tags_id.append(self.subwordset[src_tag])
        
        for tgt_tag in tgt_language_tags:
            assert tgt_tag in self.subwordset.keys()
            self.tgt_language_tags_id.append(self.subwordset[tgt_tag])

        self.envs = [lmdb.open(path, readonly=True, lock=False) for path in lmdb_path_list]
        self.env_keys = []
        for env_id in range(len(self.envs)):
            env = self.envs[env_id]
            with env.begin(write=False) as txn:
                keys = list(txn.cursor().iternext(values=False))
                for key in keys:
                    self.env_keys.append((env_id, key))

    def __len__(self):
        return len(self.env_keys)
    
    def __getitem__(self, idx):
        env_id = self.env_keys[idx][0]
        env = self.envs[env_id]
        env_key = self.env_keys[idx][1]
        with env.begin(write=False) as txn:
            value = txn.get(env_key)
            data = pickle.loads(value)
            
            src = data["src_text"].decode()
            tgt = data["tgt_text"].decode()
            idx_tgt_text = self.text_sp.EncodeAsIds(tgt.strip())
            idx_src_text = self.text_sp.EncodeAsIds(src.strip())
            assert len(idx_tgt_text) <= self.orig_max_text_len , f"length of tgt text is {len(idx_tgt_text)} > {self.orig_max_text_len}"
            assert len(idx_src_text) <= self.orig_max_text_len , f"length of src text is {len(idx_src_text)} > {self.orig_max_text_len}"
            
            bos_text = [self.src_language_tags_id[env_id]] + idx_src_text
            text_eos = idx_src_text + [self.subwordset["<eos>"]]
            src_text = bos_text + [self.subwordset["<pad>"]] * (self.max_text_len - len(bos_text))
            label_src_text = text_eos + [self.subwordset["<pad>"]] * (self.max_text_len - len(text_eos))
            bos_text = [self.tgt_language_tags_id[env_id]] + idx_tgt_text
            text_eos = idx_tgt_text + [self.subwordset["<eos>"]]
            tgt_text = bos_text + [self.subwordset["<pad>"]] * (self.max_text_len - len(bos_text))
            label_tgt_text = text_eos + [self.subwordset["<pad>"]] * (self.max_text_len - len(text_eos))

            src_img = Image.frombytes(data["src_img_mode"], data["src_img_size"], data["src_img"])
            tgt_img = Image.frombytes(data["tgt_img_mode"], data["tgt_img_size"], data["tgt_img"])
            background = Image.frombytes(data["background_mode"], data["background_size"], data["background"])
            tgt_textimg = Image.frombytes(data["tgt_textimg_mode"], data["tgt_textimg_size"], data["tgt_textimg"])

        return self.transform(src_img), self.transform(tgt_img), self.transform(background), self.transform(tgt_textimg), torch.tensor(src_text), torch.tensor(label_src_text), torch.tensor(tgt_text), torch.tensor(label_tgt_text)
