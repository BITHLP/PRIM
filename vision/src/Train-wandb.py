import torch
from torch import nn
import accelerate
from accelerate import Accelerator
from einops import rearrange
import numpy as np
import datetime
import os
import sys
sys.path.append(os.path.abspath("../../"))
import argparse
from accelerate.utils import set_seed
import sentencepiece as sp
from vision.src.VisionModel import Vision
from vision.src.Data import LMDBVisionDataset
from vision.src.utils import inverse_sqrt_lr_schedule, save_model, keep_best_models, keep_last_models, calculate_token_acc, load_config, clean_garbage
from torch.utils.data import DataLoader
import lpips


def valid(step, model, valid_dl, loss_func, accelerator):
    model.eval()
    metrics = {"loss": [], "back1_l2": [], "back2_l2": [], "code1_l2": [], "code2_l2": [], "img_l2": [], "ocr_loss": [], "tit_loss": []}
    ocr_total_token = 0
    ocr_total_acc_token = 0
    tit_total_token = 0
    tit_total_acc_token = 0
    with torch.no_grad():
        for src_img, tgt_img, background, tgt_textimg, src_text, label_src_text, tgt_text, label_tgt_text in valid_dl:
            output_dict = model(src_img, tgt_img, src_text, tgt_text)
            
            #####background decoder
            back_dict = output_dict["back_dict"]
            back_decoder_1_l2 = (back_dict["decoder_1_img"] - background).pow(2).mean()
            back_decoder_1_lpip = loss_func["back_lpip1"](back_dict["decoder_1_img"], background).mean()
            back_decoder_1_loss = back_decoder_1_l2 + 0.1*back_decoder_1_lpip

            back_decoder_2_l2 = (back_dict["decoder_2_img"] - background).pow(2).mean()
            back_decoder_2_lpip = loss_func["back_lpip2"](back_dict["decoder_2_img"], background).mean()
            back_decoder_2_loss = back_decoder_2_l2 + 0.1*back_decoder_2_lpip

            back_loss = back_decoder_1_loss + back_decoder_2_loss

            ######codebook quant decoder
            code_dict = output_dict["code_dict"]
            code_decoder_1_l2 = (code_dict["decoder_1_img"] - tgt_textimg).pow(2).mean()
            code_decoder_1_lpip = loss_func["code_lpip1"](code_dict["decoder_1_img"], tgt_textimg).mean()
            code_decoder_1_loss = code_decoder_1_l2 + 0.1*code_decoder_1_lpip

            code_decoder_2_l2 = (code_dict["decoder_2_img"] - tgt_textimg).pow(2).mean()
            code_decoder_2_lpip = loss_func["code_lpip2"](code_dict["decoder_2_img"], tgt_textimg).mean()
            code_decoder_2_loss = code_decoder_2_l2 + 0.1*code_decoder_2_lpip

            code_vq_loss = code_dict["vqloss"]
            code_loss = code_decoder_1_loss + code_decoder_2_loss + code_vq_loss

            ######img loss
            rec_img = output_dict["img"]
            img_l2 = (rec_img - tgt_img).pow(2).mean()
            img_lpip = loss_func["img_lpip"](rec_img, tgt_img).mean()
            img_loss = img_l2 + 0.1*img_lpip


            ######mt loss
            mt_dict = output_dict["mt_dict"]
            ocr_text = mt_dict["ocr_text"]
            tit_text = mt_dict["tit_text"]
            ocr_loss = loss_func["text"](rearrange(ocr_text, 'b s c -> (b s) c'), rearrange(label_src_text, 'b s -> (b s)'))
            tit_loss = loss_func["text"](rearrange(tit_text, 'b s c -> (b s) c'), rearrange(label_tgt_text, 'b s -> (b s)'))
            mt_loss = ocr_loss + tit_loss

            loss = back_loss + code_loss + img_loss + mt_loss
            
            metrics["back1_l2"].append(accelerator.gather_for_metrics(back_decoder_1_l2).mean().item())
            metrics["back2_l2"].append(accelerator.gather_for_metrics(back_decoder_2_l2).mean().item())
            metrics["code1_l2"].append(accelerator.gather_for_metrics(code_decoder_1_l2).mean().item())
            metrics["code2_l2"].append(accelerator.gather_for_metrics(code_decoder_2_l2).mean().item())
            metrics["img_l2"].append(accelerator.gather_for_metrics(img_l2).mean().item())
            metrics["ocr_loss"].append(accelerator.gather_for_metrics(ocr_loss).mean().item())
            metrics["tit_loss"].append(accelerator.gather_for_metrics(tit_loss).mean().item())
            metrics["loss"].append(accelerator.gather_for_metrics(loss).mean().item())

            # calculate ocr accuracy
            ocr_pred_tokens = accelerator.gather_for_metrics(ocr_text).argmax(-1)
            ocr_label_tokens = accelerator.gather_for_metrics(label_src_text)
            correct, total = calculate_token_acc(ocr_pred_tokens, ocr_label_tokens, model.module.multitask_encdec.padding_idx)
            ocr_total_acc_token += correct
            ocr_total_token += total

            # calculate tit accuracy
            tit_pred_tokens = accelerator.gather_for_metrics(tit_text).argmax(-1)
            tit_label_tokens = accelerator.gather_for_metrics(label_tgt_text)
            correct, total = calculate_token_acc(tit_pred_tokens, tit_label_tokens, model.module.multitask_encdec.padding_idx)
            tit_total_acc_token += correct
            tit_total_token += total
    
    ocr_token_acc = ocr_total_acc_token / ocr_total_token
    tit_token_acc = tit_total_acc_token / tit_total_token

    accelerator.log({f"valid/{key}": np.mean(metrics[key]) for key in metrics}, step=step)
    accelerator.log({"valid/ocr_acc": ocr_token_acc, "valid/tit_acc": tit_token_acc}, step=step)
    accelerator.print("{} step {}: valid loss={}, ocr_acc={}, tit_acc={}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, np.mean(metrics["loss"]), ocr_token_acc, tit_token_acc))
    model.train()
    return tit_token_acc

def train(step, model, train_dl, valid_dl, optimizer, lr_schedule, loss_func, accelerator, save_path):
    model.train()
    for src_img, tgt_img, background, tgt_textimg, src_text, label_src_text, tgt_text, label_tgt_text in train_dl:
        with accelerator.accumulate(model):
            output_dict = model(src_img, tgt_img, src_text, tgt_text)
            
            #####background decoder
            back_dict = output_dict["back_dict"]
            back_decoder_1_l2 = (back_dict["decoder_1_img"] - background).pow(2).mean()
            back_decoder_1_lpip = loss_func["back_lpip1"](back_dict["decoder_1_img"], background).mean()
            back_decoder_1_loss = back_decoder_1_l2 + 0.1*back_decoder_1_lpip

            back_decoder_2_l2 = (back_dict["decoder_2_img"] - background).pow(2).mean()
            back_decoder_2_lpip = loss_func["back_lpip2"](back_dict["decoder_2_img"], background).mean()
            back_decoder_2_loss = back_decoder_2_l2 + 0.1*back_decoder_2_lpip

            back_loss = back_decoder_1_loss + back_decoder_2_loss

            ######codebook quant decoder
            code_dict = output_dict["code_dict"]
            code_decoder_1_l2 = (code_dict["decoder_1_img"] - tgt_textimg).pow(2).mean()
            code_decoder_1_lpip = loss_func["code_lpip1"](code_dict["decoder_1_img"], tgt_textimg).mean()
            code_decoder_1_loss = code_decoder_1_l2 + 0.1*code_decoder_1_lpip

            code_decoder_2_l2 = (code_dict["decoder_2_img"] - tgt_textimg).pow(2).mean()
            code_decoder_2_lpip = loss_func["code_lpip2"](code_dict["decoder_2_img"], tgt_textimg).mean()
            code_decoder_2_loss = code_decoder_2_l2 + 0.1*code_decoder_2_lpip

            code_vq_loss = code_dict["vqloss"]
            code_loss = code_decoder_1_loss + code_decoder_2_loss + code_vq_loss

            ######img loss
            rec_img = output_dict["img"]
            img_l2 = (rec_img - tgt_img).pow(2).mean()
            img_lpip = loss_func["img_lpip"](rec_img, tgt_img).mean()
            img_loss = img_l2 + 0.1*img_lpip


            ######mt loss
            mt_dict = output_dict["mt_dict"]
            ocr_text = mt_dict["ocr_text"]
            tit_text = mt_dict["tit_text"]
            ocr_loss = loss_func["text"](rearrange(ocr_text, 'b s c -> (b s) c'), rearrange(label_src_text, 'b s -> (b s)'))
            tit_loss = loss_func["text"](rearrange(tit_text, 'b s c -> (b s) c'), rearrange(label_tgt_text, 'b s -> (b s)'))
            mt_loss = ocr_loss + tit_loss

            loss = back_loss + code_loss + img_loss + mt_loss

            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                lr_schedule.step()
                step += 1

        if accelerator.sync_gradients:
            accelerator.log({
                "train/back1_l2": back_decoder_1_l2.item(), "train/back2_l2": back_decoder_2_l2.item(),
                "train/code1_l2": code_decoder_1_l2.item(), "train/code2_l2": code_decoder_2_l2.item(),
                "train/code_vq": code_vq_loss.item(), "train/img_l2": img_l2.item(),
                "train/ocr_loss": ocr_loss.item(), "train/tit_loss": tit_loss.item(),
                "train/loss": loss.item(), "train/lr": lr_schedule.lr,
            }, step=step)

            if step % 500 == 0 and step != 0:
                accelerator.print("{} step {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step))
            if step % 1000 == 0 and step != 0:
                valid_metric = valid(step, model, valid_dl, loss_func, accelerator)
                if accelerator.is_local_main_process:
                    save_model(model, optimizer, lr_schedule, accelerator, valid_metric, save_path, "checkpoint_best{:.5f}".format(valid_metric))
                    keep_best_models(save_path, "checkpoint_best", 2, "larger")
                    if step % 5000 == 0 and step != 0:
                        save_model(model, optimizer, lr_schedule, accelerator, valid_metric, save_path, "checkpoint_last{}".format(step))
                        keep_last_models(save_path, "checkpoint_last", 4)
    accelerator.print("{} epoch end".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    clean_garbage()
    return step    

def train_loop(config_path):
    data_config, train_config, model_config = load_config(config_path)
    accelerator = Accelerator(mixed_precision="fp16", log_with="wandb", gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

    assert train_config["wandb_mode"] in ["disabled", "offline", "online"]
    accelerator.init_trackers(
        project_name=train_config["wandb_name"],
        init_kwargs={"wandb": {"mode": train_config["wandb_mode"], "config": model_config, "name": f"Vision | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}}
    )
    accelerator.print({**data_config, **train_config})

    save_path = train_config["save_checkpoint_dir"]
    if accelerator.is_local_main_process:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        os.system("cp {} {}".format(config_path, save_path))

    train_ds = LMDBVisionDataset(data_config["train_lmdb_path"], data_config["spm_path"], data_config["src_language_tags"], data_config["tgt_language_tags"], data_config["max_text_len"])
    valid_ds = LMDBVisionDataset(data_config["val_lmdb_path"], data_config["spm_path"], data_config["src_language_tags"], data_config["tgt_language_tags"], data_config["max_text_len"])
    train_dl = DataLoader(train_ds, batch_size=train_config["batch_size"], shuffle=True, num_workers=2, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=train_config["batch_size"]//2, shuffle=False)

    config = model_config
    config["mt_subwordset"] = train_ds.subwordset
    config["src_language_tags"] = data_config["src_language_tags"]
    config["tgt_language_tags"] = data_config["tgt_language_tags"]
    model = Vision(config)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9, foreach=False)
    lr_schedule = inverse_sqrt_lr_schedule(optimizer,  train_config["warmup_steps"], train_config["warmup_init_lr"], train_config["max_lr"])

    loss_func = {"back_lpip1": lpips.LPIPS(net="vgg", verbose=False).to(accelerator.device), "back_lpip2": lpips.LPIPS(net="vgg", verbose=False).to(accelerator.device),
                 "code_lpip1": lpips.LPIPS(net="vgg", verbose=False).to(accelerator.device), "code_lpip2": lpips.LPIPS(net="vgg", verbose=False).to(accelerator.device),
                 "img_lpip": lpips.LPIPS(net="vgg", verbose=False).to(accelerator.device), "text": nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=config["mt_subwordset"]["<pad>"])}

    accelerator.print(model)

    accelerator.print(f"Size of train set: {len(train_ds)}")
    accelerator.print(f"Size of valid set: {len(valid_ds)}")

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )
    step = lr_schedule.steps
    accelerator.wait_for_everyone()
    e = 0
    while(True):
        e += 1
        accelerator.print("{} epoch {} start".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
        train_step = train(step, model, train_dl, valid_dl, optimizer, lr_schedule, loss_func, accelerator, save_path)
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
