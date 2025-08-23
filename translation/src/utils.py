import torch
import os
import json
import gc

class Constant_lr_schedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr):
        self.steps = 0
        self.lr = lr
    
    def step(self):
        self.steps += 1


class warmup_lr_schedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_step=0):
        self.steps = 0
        self.d_model = d_model
        self.warmup_step = warmup_step
        self.optimizer = optimizer
        self.lr = 0.0
    
    def step(self):
        self.steps += 1
        self.lr = self.d_model**-0.5 * min(self.steps**-0.5, self.steps * self.warmup_step**-1.5)
        self.optimizer.param_groups[0]['lr'] = self.lr


class inverse_sqrt_lr_schedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_step, warmup_init_lr, max_lr):
        self.steps = 0
        self.optimizer = optimizer
        self.warmup_step = warmup_step
        self.warmup_init_lr = warmup_init_lr
        self.max_lr = max_lr
        self.lrs = torch.linspace(self.warmup_init_lr, self.max_lr, self.warmup_step)
        self.lr = 0.0
    
    def step(self):
        self.steps += 1
        if self.steps < self.warmup_step:
            self.lr = self.lrs[self.steps]
        else:
            decay_factor = self.max_lr * self.warmup_step**0.5
            self.lr = decay_factor * self.steps**-0.5
        self.optimizer.param_groups[0]['lr'] = self.lr


def save_model(model, optimizer, lr_schedule, accelerator, metric, path, name):
    unwrap_model = accelerator.unwrap_model(model)
    torch.save({
        "model_state" : unwrap_model.state_dict(),
        "metric": metric
        }, os.path.join(path, name+".pt"))
    accelerator.print("save checkpoint at {}".format(os.path.join(path, name+".pt")))

def keep_best_models(save_path, prefix="checkpoint_best", num=5, better_metric="larger"):
    assert better_metric in ["larger", "smaller"]
    reverse = None
    if better_metric == "larger":
        reverse = True
    elif better_metric == "smaller":
        reverse = False
    else:
        raise ValueError("Unknown better metric: {}".format(better_metric))

    model_files = [f for f in os.listdir(save_path) if prefix in f]
    metric_values = [float(f.replace(".pt", "").replace(prefix, "")) for f in model_files]

    sorted_indices = sorted(range(len(metric_values)), key=lambda k: metric_values[k], reverse=reverse)
    models_to_keep = sorted_indices[:num]
    for i, model_file in enumerate(model_files):
        if i not in models_to_keep:
            os.remove(os.path.join(save_path, model_file))

def keep_last_models(save_path, prefix="checkpoint_last", num=5):
    model_files = [f for f in os.listdir(save_path) if prefix in f]
    metric_values = [float(f.replace(".pt", "").replace(prefix, "")) for f in model_files]

    sorted_indices = sorted(range(len(metric_values)), key=lambda k: metric_values[k], reverse=True)
    models_to_keep = sorted_indices[:num]
    for i, model_file in enumerate(model_files):
        if i not in models_to_keep:
            os.remove(os.path.join(save_path, model_file))

def calculate_token_acc(pred, tgt, padding_id=None):
    if padding_id is None:
        total = pred.numel()
        correct = (pred == tgt).sum()
    else:
        mask = tgt != padding_id
        total = mask.sum()
        correct = (pred == tgt)[mask].sum()
    return correct, total

def load_config(config_path):
    json_file = open(config_path)
    json_dict = json.load(json_file)
    data = json_dict["data"]
    train = json_dict["train"]
    model = json_dict["model"]
    return data, train, model

def clean_garbage():
    gc.collect()
    torch.cuda.empty_cache()
