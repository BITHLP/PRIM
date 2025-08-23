export CUDA_VISIBLE_DEVICES=5

config=/data1/yztian/VisTrans_models/IIMT30k/trans/checkpoint_best0.80999.pt
checkpoint=/data1/yztian/VisTrans_models/IIMT30k/trans/trans-iimt30k.json


# infer
for font in Arial Calibri TimesNewRoman; do
    python infer.py --config ${config} --checkpoint ${checkpoint} \
        --batch_size 16 --lang_tag "<de>" \
        --input_img_dir /data2/yztian/IIMT30k/${font}/test_flickr/en/image \
        --output_img_dir /data1/yztian/ACL2025/VisTrans_IIMT30k/${font}/test/ende/image
done

for font in Arial Calibri TimesNewRoman; do
    python infer.py --config ${config} --checkpoint ${checkpoint} \
        --batch_size 16 --lang_tag "<en>" \
        --input_img_dir /data2/yztian/IIMT30k/${font}/test_flickr/de/image \
        --output_img_dir /data1/yztian/ACL2025/VisTrans_IIMT30k/${font}/test/deen/image
done
