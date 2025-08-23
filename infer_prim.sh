export CUDA_VISIBLE_DEVICES=0

config=/data1/yztian/ACL2025/VisTrans_models/trans/trans.json
checkpoint=/data1/yztian/ACL2025/VisTrans_models/trans/checkpoint_best0.87278.pt

system_name=sat2


# infer
for l in de fr cs ru ro; do
    python infer.py --config ${config} --checkpoint ${checkpoint} \
        --batch_size 128 --lang_tag "<"${l}">" \
        --input_img_dir /data1/yztian/PRIM/Test/en_img \
        --output_img_dir /data1/yztian/ACL2025/VisTrans/Test/${system_name}/${l}_img
done
