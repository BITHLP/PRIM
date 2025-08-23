export CUDA_VISIBLE_DEVICES=0,1

# accelerate launch --main_process_port 3190 ../src/Train-wandb.py --config ../configs/vision.json

accelerate launch --main_process_port 3190 ../src/Train-wandb.py --config ../configs/vision-iimt30k.json
