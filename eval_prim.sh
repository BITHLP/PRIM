export CUDA_VISIBLE_DEVICES=0

system_name=sat2

# ocr
for l in de fr cs ru ro; do
    python ocr.py --tgt_l ${l} --input_img_dir /data1/yztian/ACL2025/VisTrans/Test/${system_name}/${l}_img --output_ocr_file /data1/yztian/ACL2025/VisTrans/Test/${system_name}/ocr.${l}
done


# bleu
echo "google & gpt4 translation bleu"
for l in de fr cs ru ro; do
    sacrebleu /data1/yztian/PRIM/Test/google/test_google.${l} /data1/yztian/PRIM/Test/gpt4/test_gpt4.${l} -i /data1/yztian/ACL2025/VisTrans/Test/${system_name}/ocr.${l} -w 2
done


# comet
echo "google translation comet"
for l in de fr cs ru ro; do
    comet-score -s /data1/yztian/PRIM/Test/test.en -t /data1/yztian/ACL2025/VisTrans/Test/${system_name}/ocr.${l} -r /data1/yztian/PRIM/Test/google/test_google.${l} --only_system --quiet
done

echo "gpt4 translation comet"
for l in de fr cs ru ro; do
    comet-score -s /data1/yztian/PRIM/Test/test.en -t /data1/yztian/ACL2025/VisTrans/Test/${system_name}/ocr.${l} -r /data1/yztian/PRIM/Test/gpt4/test_gpt4.${l} --only_system --quiet
done


# fid
echo "google translation fid"
for l in de fr cs ru ro; do
    python -m pytorch_fid /data1/yztian/ACL2025/VisTrans/Test/${system_name}/${l}_img /data1/yztian/PRIM/Test/google/${l}_google
done

echo "gpt4 translation fid"
for l in de fr cs ru ro; do
    python -m pytorch_fid /data1/yztian/ACL2025/VisTrans/Test/${system_name}/${l}_img /data1/yztian/PRIM/Test/gpt4/${l}_gpt4
done
