import os
import sys
import cv2

from layout import LayoutAnalyzer, visualize
import numpy as np

from text import TextSystem

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

from tools.infer.text.config import parse_args
from tools.infer.text.utils import crop_text_region

def sharpen_image(image, save_path=None):
    # 定义锐化卷积核
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    
    # 应用卷积核进行图像锐化
    sharpened = cv2.filter2D(image, -1, kernel)
    if save_path:
        cv2.imwrite(save_path, sharpened)
    return sharpened

def add_padding(image, padding_size, padding_color=(0, 0, 0)):
    """
    给图像添加填充
    :param image: 输入图像
    :param padding_size: 填充大小，可以是单个整数（四周相同大小）或包含四个整数的元组（分别表示上、下、左、右的填充大小）
    :param padding_color: 填充颜色，默认为黑色
    :return: 添加填充后的图像
    """
    if isinstance(padding_size, int):
        top, bottom, left, right = padding_size, padding_size, padding_size, padding_size
    else:
        top, bottom, left, right = padding_size

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return padded_image

def sort_words_by_poly(words, polys):
    from functools import cmp_to_key
    """
    根据文本框的位置信息对识别结果进行排序
    :param words: 识别结果
    :param polys: 文本框位置信息
    :return: 排序后的识别结果

    """
    def compare(x, y):
        dist1 = y[1][3][1] - x[1][0][1]
        dist2 = x[1][3][1] - y[1][0][1]
        if abs(dist1 - dist2) < x[1][3][1] - x[1][0][1] or abs(dist1 - dist2) < y[1][3][1] - y[1][0][1]:
            if x[1][0][0] < y[1][0][0]:
                return -1
            elif x[1][0][0] == y[1][0][0]:
                return 0
            else:
                return 1
        else:
            if x[1][0][1] < y[1][0][1]:
                return -1
            elif x[1][0][1] == y[1][0][1]:
                return 0
            else:
                return 1
    tmp = sorted(zip(words, polys), key=cmp_to_key(compare))
    return [item[0][0] for item in tmp]

do_recovery = True

def main():
    args = parse_args()
    text_system = TextSystem(args)
    layout_analyzer = LayoutAnalyzer(ckpt_load_path="/home/cjh/mindocr/yolov8n.ckpt")
    image_path = "/home/cjh/mindocr/PMC4958442_00003.jpg"
    results = layout_analyzer.infer(image_path)

    # crop text regions
    image = cv2.imread(image_path)
    h_ori, w_ori = image.shape[:2]
    crops = []
    category_dict = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}
    text_results = []
    for i in range(len(results)):
        category_id = results[i]['category_id']
        left, top, w, h = results[i]['bbox']
        right = left + w
        bottom = top + h
        cropped_img = image[int(top):int(bottom), int(left):int(right)]
        # cropped_img = crop_text_region(image, polygon)
        padding_size = 10  # 可以根据需要调整填充大小
        padding_color = (255, 255, 255)  # 白色填充
        cropped_img = add_padding(cropped_img, padding_size, padding_color)
        save_path = f"/home/cjh/mindocr/cjh_result/output_00000_crop_{i}.jpg"
        cv2.imwrite(save_path, cropped_img)
        # cropped_img = sharpen_image(cropped_img, "/home/cjh/mindocr/output_00000_crop_sharpen.jpg")
        crops.append(cropped_img)
        rec_res_all_crops = text_system(cropped_img, do_visualize=False)
        output = sort_words_by_poly(rec_res_all_crops[1], rec_res_all_crops[0])
        text_results.append({"category_id": category_id, "bbox": [left, top, w, h], "text": " ".join(output)})
    print(text_results)

if __name__ == "__main__":
    main()