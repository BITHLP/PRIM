cat /data2/yztian/IIMT30k/TimesNewRoman/train/de/subtitle.txt /data2/yztian/IIMT30k/TimesNewRoman/train/en/subtitle.txt > /data2/yztian/iimt30k_train.total
spm_train --unk_id=0 --bos_id=-1 --eos_id=1 --eos_piece "<eos>" --pad_id=2 --input=/data2/yztian/iimt30k_train.total --model_prefix=total-10kbpe --vocab_size=10000 --model_type=bpe --user_defined_symbols "<en>","<de>"
spm_train --unk_id=0 --bos_id=1 --eos_id=2 --bos_piece "<bos>" --eos_piece "<eos>" --pad_id=3 --input=/data2/yztian/iimt30k_train.total --model_prefix=total-char --model_type=char
