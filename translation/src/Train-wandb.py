import os
import sys
sys.path.append(os.path.abspath("../../"))
import torch
from torch import nn
import accelerate
from accelerate import Accelerator
from einops import rearrange
import numpy as np
import datetime
import argparse
from accelerate.utils import set_seed
import sentencepiece as sp
from translation.src.TransModel import Translation
from translation.src.Data import LMDBTransDataset
from vision.src.VisionModel import Vision
from translation.src.utils import inverse_sqrt_lr_schedule, save_model, keep_best_models, keep_last_models, calculate_token_acc, load_config, clean_garbage
from torch.utils.data import DataLoader


def valid(step, trans_model, vision_model, valid_dl, loss_func, accelerator):
    trans_model.eval()
    metrics = {"loss": [], "code_loss": [], "sub_text_loss": []}
    total_code = 0
    total_acc_code = 0

    tit_total_token = 0
    tit_total_acc_token = 0

    sub_text_total_token = 0
    sub_text_total_acc_token = 0
    with torch.no_grad():
        for src_img, tgt_text, label_tgt_text, code, label_code, sub_text, label_sub_text in valid_dl:
            # inference [tgt_text_hidden] with [src_img, tgt_text] by vision model
            if isinstance(vision_model, torch.nn.parallel.DistributedDataParallel):
                text_hidden_dict = vision_model.module.inference_tit(src_img, tgt_text)
            else:
                text_hidden_dict = vision_model.inference_tit(src_img, tgt_text)
            
            output_dict = trans_model(text_hidden_dict["hidden_text"], code, sub_text, text_hidden_dict["padding_mask"])
            output_code = output_dict["code"]
            code_loss = loss_func["code"](rearrange(output_code, 'b s c -> (b s) c'), rearrange(label_code, 'b s -> (b s)'))
            output_sub_text = output_dict["sub_text"]
            sub_text_loss = loss_func["sub_text"](rearrange(output_sub_text, 'b s c -> (b s) c'), rearrange(label_sub_text, 'b s -> (b s)'))

            loss = code_loss + sub_text_loss

            metrics["loss"].append(accelerator.gather_for_metrics(loss).mean().item())
            metrics["code_loss"].append(accelerator.gather_for_metrics(code_loss).mean().item())
            metrics["sub_text_loss"].append(accelerator.gather_for_metrics(sub_text_loss).mean().item())

            # calculate code accuracy
            pred_codes = accelerator.gather_for_metrics(output_code).argmax(-1)
            label_codes = accelerator.gather_for_metrics(label_code)
            correct, total = calculate_token_acc(pred_codes, label_codes)
            total_acc_code += correct
            total_code += total

            # calculate tit accuracy
            tit_pred_tokens = accelerator.gather_for_metrics(text_hidden_dict["output_text"]).argmax(-1)
            tit_label_tokens = accelerator.gather_for_metrics(label_tgt_text)
            if isinstance(vision_model, torch.nn.parallel.DistributedDataParallel):
                pad_idx = vision_model.module.multitask_encdec.padding_idx
            else:
                pad_idx = vision_model.multitask_encdec.padding_idx
            correct, total = calculate_token_acc(tit_pred_tokens, tit_label_tokens, pad_idx)
            tit_total_acc_token += correct
            tit_total_token += total

            # calculate sub text accuracy
            pred_sub_texts = accelerator.gather_for_metrics(output_sub_text).argmax(-1)
            label_sub_texts = accelerator.gather_for_metrics(label_sub_text)
            if isinstance(trans_model, torch.nn.parallel.DistributedDataParallel):
                pad_idx = trans_model.module.sub_text_padding
            else:
                pad_idx = trans_model.sub_text_padding
            correct, total = calculate_token_acc(pred_sub_texts, label_sub_texts, pad_idx)
            sub_text_total_acc_token += correct
            sub_text_total_token += total

    code_acc = total_acc_code / total_code
    tit_acc = tit_total_acc_token / tit_total_token
    sub_text_acc = sub_text_total_acc_token / sub_text_total_token

    accelerator.log({
        "valid/loss": np.mean(metrics["loss"]), "valid/code_loss": np.mean(metrics["code_loss"]), "valid/sub_text_loss": np.mean(metrics["sub_text_loss"]),
        "valid/code_acc": code_acc, "valid/tit_acc": tit_acc, "valid/sub_text_acc": sub_text_acc,
        }, step=step)

    accelerator.print("{} step {}: valid loss={}, code_acc={}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, np.mean(metrics["loss"]), code_acc))
    trans_model.train()
    return code_acc

def train(step, trans_model, vision_model, train_dl, valid_dl, optimizer, lr_schedule, loss_func, accelerator, save_path):
    trans_model.train()
    for src_img, tgt_text, _, code, label_code, sub_text, label_sub_text in train_dl:
        # inference [tgt_text_hidden] with [src_img, tgt_text] by vision model
        with torch.no_grad():
            if isinstance(vision_model, torch.nn.parallel.DistributedDataParallel):
                text_hidden_dict = vision_model.module.inference_tit(src_img, tgt_text)
            else:
                text_hidden_dict = vision_model.inference_tit(src_img, tgt_text)
        with accelerator.accumulate(trans_model):
            output_dict = trans_model(text_hidden_dict["hidden_text"], code, sub_text, text_hidden_dict["padding_mask"])
            output_code = output_dict["code"]
            code_loss = loss_func["code"](rearrange(output_code, 'b s c -> (b s) c'), rearrange(label_code, 'b s -> (b s)'))
            output_sub_text = output_dict["sub_text"]
            sub_text_loss = loss_func["sub_text"](rearrange(output_sub_text, 'b s c -> (b s) c'), rearrange(label_sub_text, 'b s -> (b s)'))

            loss = code_loss + sub_text_loss
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                lr_schedule.step()
                step += 1

        if accelerator.sync_gradients:
            accelerator.log({"train/loss": loss.item(), "train/code_loss": code_loss.item(), "train/sub_text_loss": sub_text_loss.item(), "train/lr": lr_schedule.lr}, step=step)
            if step % 500 == 0 and step != 0:
                accelerator.print("{} step {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step))
            if step % 1000 == 0 and step != 0:
                valid_metric = valid(step, trans_model, vision_model, valid_dl, loss_func, accelerator)
                if accelerator.is_local_main_process:
                    save_model(trans_model, optimizer, lr_schedule, accelerator, valid_metric, save_path, "checkpoint_best{:.5f}".format(valid_metric))
                    keep_best_models(save_path, "checkpoint_best", 2, "larger")
                    if step % 5000 == 0 and step != 0:
                        save_model(trans_model, optimizer, lr_schedule, accelerator, valid_metric, save_path, "checkpoint_last{}".format(step))
                        keep_last_models(save_path, "checkpoint_last", 4)
    accelerator.print("{} epoch end".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    clean_garbage()
    return step    

def train_loop(config_path):
    data_config, train_config, trans_config = load_config(config_path)
    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb", gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

    assert train_config["wandb_mode"] in ["disabled", "offline", "online"]
    accelerator.init_trackers(
        project_name=train_config["wandb_name"],
        init_kwargs={"wandb": {"mode": train_config["wandb_mode"], "config": trans_config, "name": f"Translation | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}}
    )
    accelerator.print({**data_config, **train_config})
    
    save_path = train_config["save_checkpoint_dir"]
    if accelerator.is_local_main_process:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        os.system("cp {} {}".format(config_path, save_path))

    train_ds = LMDBTransDataset(data_config["train_lmdb_path"], data_config["spm_path"], data_config["subspm_path"], data_config["train_code_file_path"], trans_config["code_bos"], trans_config["code_eos"], 
                                data_config["src_language_tags"], data_config["tgt_language_tags"], data_config["max_text_len"], data_config["max_subtext_len"], trans_config["group_size"])
    valid_ds = LMDBTransDataset(data_config["val_lmdb_path"], data_config["spm_path"], data_config["subspm_path"], data_config["val_code_file_path"], trans_config["code_bos"], trans_config["code_eos"], 
                                data_config["src_language_tags"], data_config["tgt_language_tags"], data_config["max_text_len"], data_config["max_subtext_len"], trans_config["group_size"])
    train_dl = DataLoader(train_ds, batch_size=train_config["batch_size"], shuffle=True, num_workers=2, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=train_config["batch_size"]//2, shuffle=False)

    vision_config_path = train_config["vision_config"]
    vision_ckpt_path = train_config["vision_ckpt"]

    _, _, vision_config = load_config(vision_config_path)
    vision_config["mt_subwordset"] = train_ds.subwordset

    vision_config["src_language_tags"] = data_config["src_language_tags"]
    vision_config["tgt_language_tags"] = data_config["tgt_language_tags"]

    assert trans_config["code_num"] == vision_config["codebook_size"] + 2 # add <bos> and <eos>
    assert vision_config["codebook_size"] <= trans_config["code_bos"]
    assert vision_config["codebook_size"] <= trans_config["code_eos"]
    trans_config["vision_d_model"] = vision_config["mt_d_model"]
    trans_config["sub_subwordset"] = train_ds.sub_subwordset
    trans_config["max_seq_len"] = train_ds.max_subtext_len
    trans_model = Translation(trans_config)

    vision_ckpt = torch.load(vision_ckpt_path, map_location=accelerator.device)
    vision_model = Vision(vision_config)
    vision_model.load_state_dict(vision_ckpt["model_state"])

    optimizer = torch.optim.AdamW(params=trans_model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9, foreach=False)
    lr_schedule = inverse_sqrt_lr_schedule(optimizer,  train_config["warmup_steps"], train_config["warmup_init_lr"], train_config["max_lr"])

    loss_func = {"code": nn.CrossEntropyLoss(label_smoothing=0.1), "sub_text": nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=train_ds.sub_subwordset["<pad>"])}

    accelerator.print(trans_model)

    accelerator.print(f"Size of train set: {len(train_ds)}")
    accelerator.print(f"Size of valid set: {len(valid_ds)}")

    trans_model, vision_model, optimizer, train_dl, valid_dl = accelerator.prepare(
        trans_model, vision_model, optimizer, train_dl, valid_dl
    )
    vision_model.eval()
    step = lr_schedule.steps
    accelerator.wait_for_everyone()
    e = 0
    while(True):
        e += 1
        accelerator.print("{} epoch {} start".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
        train_step = train(step, trans_model, vision_model, train_dl, valid_dl, optimizer, lr_schedule, loss_func, accelerator, save_path)
        step = train_step
        if step >= train_config["max_update_step"]:
            break
        if train_config["epoch"] != -1 and e >= train_config["epoch"]:
            break
    accelerator.print("{} training end".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    accelerator.end_training()


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()
    config = args.config
    train_loop(config)
