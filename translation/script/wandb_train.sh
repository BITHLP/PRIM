export CUDA_VISIBLE_DEVICES=0,1

# accelerate launch --main_process_port 4190 ../src/Train-wandb.py --config ../configs/trans.json

accelerate launch --main_process_port 4190 ../src/Train-wandb.py --config ../configs/trans-iimt30k.json