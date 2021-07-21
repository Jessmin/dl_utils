from paddleocr import PaddleOCR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i', dest='input_dir', type=str, required=True)
parser.add_argument('--o', dest='output_dir', type=str, required=True)
args = parser.parse_args()
ocr = PaddleOCR(
    det_model_dir=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/det',
    rec_model_dir=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/rec',
    rec_char_dict_path=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/ppocr_keys_v1.txt',
    cls_model_dir=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/cls',
    use_angle_cls=True,
    max_text_length=15,
    drop_score=0.5,
    det_db_unclip_ratio=2.0,
    lang="ch")


def predict(img):
    ocr_data = ocr.ocr(img, cls=True)
    return ocr_data


if __name__ == '__main__':
    import glob
    import cv2
    import os
    import shutil

    files = glob.glob(f'{args.input_dir}/*.png')
    files.extend(glob.glob(f'{args.input_dir}/*.jpeg'))
    files.extend(glob.glob(f'{args.input_dir}/*.jpg'))
    files.extend(glob.glob(f'{args.input_dir}/*.webp'))
    if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
    for file in files:
        _, filename = os.path.split(file)
        filename, _ = os.path.splitext(filename)
        output_filename = f'{args.output_dir}/{filename}.png.txt'
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        img = cv2.imread(file)
        ocr_data = predict(img)
        with open(output_filename, 'a') as f:
            for data in ocr_data:
                box = data[0]
                txt = data[1][0]
                confidence = data[1][1]
                line = f"{', '.join([', '.join(str(int(m)) for m in x) for x in box])},{confidence},{txt}\n".replace(
                    ' ', '')
                f.write(line)
        del ocr_data
