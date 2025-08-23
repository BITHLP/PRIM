export CUDA_VISIBLE_DEVICES=0

## PRIM
# tgt_lang=de # de fr cs ru ro
# config=/data1/yztian/ACL2025/VisTrans_models/vision/vision.json
# checkpoint=/data1/yztian/ACL2025/VisTrans_models/vision/checkpoint_best0.59790.pt

# for tgt_lang in de fr cs ru ro; do
#     python ../src/infer_code.py \
#         --config ${config} \
#         --checkpoint ${checkpoint} \
#         --input_lmdb /data2/yztian/MTedIIMT/subset_en_${tgt_lang}_lmdb/train \
#         --output_code /data2/yztian/vistrans_codes/train.${tgt_lang}

#     python ../src/infer_code.py \
#         --config ${config} \
#         --checkpoint ${checkpoint} \
#         --input_lmdb /data2/yztian/MTedIIMT/subset_en_${tgt_lang}_lmdb/val \
#         --output_code /data2/yztian/vistrans_codes/val.${tgt_lang}
# done

## IIMT30k
config=/data1/yztian/ACL2025/VisTrans_models_IIMT30k/vision/vision-iimt30k.json
checkpoint=/data1/yztian/ACL2025/VisTrans_models_IIMT30k/vision/checkpoint_best0.65922.pt

for tgt_lang in ende deen; do
    python ../src/infer_code.py \
        --config ${config} \
        --checkpoint ${checkpoint} \
        --input_lmdb /data2/yztian/IIMT30k_train_${tgt_lang} \
        --output_code /data2/yztian/VisTrans_codes_IIMT30k/train.${tgt_lang}

    python ../src/infer_code.py \
        --config ${config} \
        --checkpoint ${checkpoint} \
        --input_lmdb /data2/yztian/IIMT30k_val_${tgt_lang} \
        --output_code /data2/yztian/VisTrans_codes_IIMT30k/val.${tgt_lang}
done
