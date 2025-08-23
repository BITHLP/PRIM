export CUDA_VISIBLE_DEVICES=5

for font in Arial Calibri TimesNewRoman; do
    python ocr.py --tgt_l de --input_img_dir /data1/yztian/ACL2025/VisTrans_IIMT30k/${font}/test/ende/image --output_ocr_file /data1/yztian/ACL2025/VisTrans_IIMT30k/${font}/test/ende/ocr.de
done

echo "=====IIMT30k En-De BLEU====="
rm /data1/yztian/VisTrans_de.total
cat /data1/yztian/ACL2025/VisTrans_IIMT30k/Arial/test/ende/ocr.de /data1/yztian/ACL2025/VisTrans_IIMT30k/Calibri/test/ende/ocr.de /data1/yztian/ACL2025/VisTrans_IIMT30k/TimesNewRoman/test/ende/ocr.de > /data1/yztian/VisTrans_de.total

rm /data1/yztian/IIMT30kRef_de.total
cat /data2/yztian/IIMT30k/Arial/test_flickr/de/subtitle.txt /data2/yztian/IIMT30k/Calibri/test_flickr/de/subtitle.txt /data2/yztian/IIMT30k/TimesNewRoman/test_flickr/de/subtitle.txt > /data1/yztian/IIMT30kRef_de.total

cat /data1/yztian/VisTrans_de.total | sacrebleu /data1/yztian/IIMT30kRef_de.total


for font in Arial Calibri TimesNewRoman; do
    python ocr.py --tgt_l en --input_img_dir /data1/yztian/ACL2025/VisTrans_IIMT30k/${font}/test/deen/image --output_ocr_file /data1/yztian/ACL2025/VisTrans_IIMT30k/${font}/test/deen/ocr.en
done

echo "=====IIMT30k De-En BLEU====="
rm /data1/yztian/VisTrans_en.total
cat /data1/yztian/ACL2025/VisTrans_IIMT30k/Arial/test/deen/ocr.en /data1/yztian/ACL2025/VisTrans_IIMT30k/Calibri/test/deen/ocr.en /data1/yztian/ACL2025/VisTrans_IIMT30k/TimesNewRoman/test/deen/ocr.en > /data1/yztian/VisTrans_en.total

rm /data1/yztian/IIMT30kRef_en.total
cat /data2/yztian/IIMT30k/Arial/test_flickr/en/subtitle.txt /data2/yztian/IIMT30k/Calibri/test_flickr/en/subtitle.txt /data2/yztian/IIMT30k/TimesNewRoman/test_flickr/en/subtitle.txt > /data1/yztian/IIMT30kRef_en.total

cat /data1/yztian/VisTrans_en.total | sacrebleu /data1/yztian/IIMT30kRef_en.total
