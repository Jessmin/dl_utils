from paddleocr import PaddleOCR

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
    drop_score=0.6,
    det_db_unclip_ratio=2.0,
    lang="ch")

def predict(img):
    ocr_data = ocr.ocr(img)
    txts = [x[1][0] for x in ocr_data]
    bboxes = [x[0] for x in ocr_data]
    return bboxes, txts