import torch
import os
import os
import sys
sys.path.append(os.path.abspath("../../"))
from vision.src.VisionModel import Vision
from vision.src.utils import load_config
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sentencepiece as sp
import lmdb
import pickle
import argparse

class TestDataset(Dataset):
    def __init__(self, lmdb_path, spm):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
        ])
        self.text_sp = spm
        self.subwordset = {}
        self.vocab_size = self.text_sp.get_piece_size()
        for i in range(self.vocab_size):
            self.subwordset[self.text_sp.id_to_piece(i)] = i
        
        self.env_keys = []
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            self.env_keys = list(txn.cursor().iternext(values=False))


    def __len__(self):
        return len(self.env_keys)

    def __getitem__(self, idx):
        key = self.env_keys[idx]
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
            data = pickle.loads(value)
            tgt_img = Image.frombytes(data["tgt_img_mode"], data["tgt_img_size"], data["tgt_img"])

        return self.transform(tgt_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_lmdb", type=str, required=True)
    parser.add_argument("--output_code", type=str, required=True)
    args = parser.parse_args()

    data_config, _, model_config = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    spm = sp.SentencePieceProcessor(data_config["spm_path"])

    output_code_dir = os.path.dirname(args.output_code)
    if not os.path.exists(output_code_dir):
        os.makedirs(output_code_dir)
    code_file = open(args.output_code, "w")

    ds = TestDataset(args.input_lmdb, spm)
    dl = DataLoader(ds, 1024, shuffle=False, num_workers=2)

    model_config["mt_subwordset"] = ds.subwordset
    model_config["src_language_tags"] = data_config["src_language_tags"]
    model_config["tgt_language_tags"] = data_config["tgt_language_tags"]
    model = Vision(model_config)
    model.load_state_dict(ckpt["model_state"])

    model.eval().cuda()

    with torch.no_grad():
        for img in dl:
            codes = model.inference_quant_code(img.cuda())["code"]
            for code in codes:
                code_file.write(" ".join([str(c) for c in code.tolist()])+"\n")
