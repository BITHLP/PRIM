import easyocr
import os
import argparse

def merge_ocr_results(results):
    sorted_results = sorted(results, key=lambda x: (x[0][0][0], x[0][0][1]))
    merged_text = ' '.join(result[1] for result in sorted_results)

    return merged_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_l", type=str, required=True)
    parser.add_argument("--input_img_dir", type=str, required=True)
    parser.add_argument("--output_ocr_file", type=str, required=True)
    args = parser.parse_args()

    reader = easyocr.Reader([args.tgt_l])

    img_dir = args.input_img_dir
    result_file = args.output_ocr_file

    img_list = sorted(os.listdir(img_dir), key=lambda x: int(x.split(".")[0]))
    result_file = open(result_file, "w")
    for img in img_list:
        result = reader.readtext(os.path.join(img_dir, img))
        result_file.write(merge_ocr_results(result))
        result_file.write("\n")
        result_file.flush()
    result_file.close()
