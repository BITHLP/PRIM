import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import lmdb
import pickle


def main(iimt30k_path, output_ende_lmdb_path, output_deen_lmdb_path, subset):
    assert subset in ["train", "val"]
    arial_path = os.path.join(iimt30k_path, "Arial", subset)
    calibri_path = os.path.join(iimt30k_path, "Calibri", subset)
    tnr_path = os.path.join(iimt30k_path, "TimesNewRoman", subset)

    if not os.path.exists(output_ende_lmdb_path):
        os.makedirs(output_ende_lmdb_path)
    
    background_list = []
    de_img_list = []
    de_textimg_list = []
    en_img_list = []
    en_textimg_list = []
    de_text_list = []
    en_text_list = []

    for path in [arial_path, calibri_path, tnr_path]: # Select the subsets in IIMT30k
        background_path = os.path.join(path, "background")
        l = sorted(os.listdir(background_path), key=lambda x: int(x.split(".")[0]))
        for ll in l:
            background_list.append(os.path.join(background_path, ll))
        
        de_img_path = os.path.join(path, "de", "image")
        l = sorted(os.listdir(de_img_path), key=lambda x: int(x.split(".")[0]))
        for ll in l:
            de_img_list.append(os.path.join(de_img_path, ll))
        
        de_textimg_path = os.path.join(path, "de", "text")
        l = sorted(os.listdir(de_textimg_path), key=lambda x: int(x.split(".")[0]))
        for ll in l:
            de_textimg_list.append(os.path.join(de_textimg_path, ll))
        
        en_img_path = os.path.join(path, "en", "image")
        l = sorted(os.listdir(en_img_path), key=lambda x: int(x.split(".")[0]))
        for ll in l:
            en_img_list.append(os.path.join(en_img_path, ll))
        
        en_textimg_path = os.path.join(path, "en", "text")
        l = sorted(os.listdir(en_textimg_path), key=lambda x: int(x.split(".")[0]))
        for ll in l:
            en_textimg_list.append(os.path.join(en_textimg_path, ll))    

        de_text_file = open(os.path.join(path, "de", "subtitle.txt"), "r")
        for l in de_text_file:
            de_text_list.append(l.strip())
        
        en_text_file = open(os.path.join(path, "en", "subtitle.txt"), "r")
        for l in en_text_file:
            en_text_list.append(l.strip())

    map_size = 1099511627776 * 2
    ende_env = lmdb.open(output_ende_lmdb_path, map_size=int(map_size))
    deen_env = lmdb.open(output_deen_lmdb_path, map_size=int(map_size))
    idx = 0
    with ende_env.begin(write=True) as ende_txn, deen_env.begin(write=True) as deen_txn:
        for back_path, de_img_path, de_textimg_path, en_img_path, en_textimg_path, de_text, en_text in zip(background_list, de_img_list, de_textimg_list, en_img_list, en_textimg_list, de_text_list, en_text_list):

            back_img = Image.open(back_path)
            de_img = Image.open(de_img_path)
            de_textimg = Image.open(de_textimg_path)
            en_img = Image.open(en_img_path)
            en_textimg = Image.open(en_textimg_path)
            key = f"{idx:08}".encode()

            ende_value = pickle.dumps({
                "background": back_img.tobytes(),
                "background_mode": back_img.mode,
                "background_size": back_img.size,
                "src_img": en_img.tobytes(),
                "src_img_mode": en_img.mode,
                "src_img_size": en_img.size,
                "tgt_img": de_img.tobytes(),
                "tgt_img_mode": de_img.mode,
                "tgt_img_size": de_img.size,
                "tgt_textimg": de_textimg.tobytes(),
                "tgt_textimg_mode": de_textimg.mode,
                "tgt_textimg_size": de_textimg.size,
                "src_text": en_text.encode(),
                "tgt_text": de_text.encode()
            })
            ende_txn.put(key, ende_value)

            deen_value = pickle.dumps({
                "background": back_img.tobytes(),
                "background_mode": back_img.mode,
                "background_size": back_img.size,
                "src_img": de_img.tobytes(),
                "src_img_mode": de_img.mode,
                "src_img_size": de_img.size,
                "tgt_img": en_img.tobytes(),
                "tgt_img_mode": en_img.mode,
                "tgt_img_size": en_img.size,
                "tgt_textimg": en_textimg.tobytes(),
                "tgt_textimg_mode": en_textimg.mode,
                "tgt_textimg_size": en_textimg.size,
                "src_text": de_text.encode(),
                "tgt_text": en_text.encode()
            })
            deen_txn.put(key, deen_value)
            idx += 1

if __name__ == "__main__":
    main("/data2/yztian/IIMT30k", "/data2/yztian/IIMT30k_val_ende", "/data2/yztian/IIMT30k_val_deen", "val")
    main("/data2/yztian/IIMT30k", "/data2/yztian/IIMT30k_train_ende", "/data2/yztian/IIMT30k_train_deen", "train")
