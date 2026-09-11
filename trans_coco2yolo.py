"""
本脚本有两个功能：
1.根据train.json标签文件将coco数据集标注信息转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据json标签文件，生成对应类别->索引json文件(cowboyoutfits_classes.json)和names标签(cowboyoutfits_label.names)
"""

import os
import json
import csv
import shutil
from tqdm import tqdm
import random


# 数据集根目录
dataset_root = "./cowboyoutfits"
images_dir = os.path.join(dataset_root, "images")
train_json_path = os.path.join(dataset_root, "train.json")
test_csv_path = os.path.join(dataset_root, "test.csv")

assert os.path.exists(images_dir), "images path not exist..."
assert os.path.exists(train_json_path), "train_json_path not exist..."
assert os.path.exists(test_csv_path), "test csv file not exist..."


save_file_root = "./cowboyoutfits_yolo_dataset"
val_ratio = 0.01
random.seed(42)

if os.path.exists(save_file_root) is False:
    os.makedirs(save_file_root, exist_ok=True)

def build_image_dict(data):
    image_dict = {}
    for image in data["images"]:
        image_id = str(image["id"])
        image_dict[image_id] = image

    return image_dict


def build_annotation_dict(data):
    annotation_dict = {}
    for annotation in data["annotations"]:
        image_id = str(annotation["image_id"])
        # 一个image_id对应几个annotation，所以是个列表字典
        if image_id not in annotation_dict:
            annotation_dict[image_id] = []
        annotation_dict[image_id].append(annotation)

    return annotation_dict


def translate_info(data, save_root):
    """
    将json标注转换成YOLO格式，
    并将对应图片划分成train和val
    train: 99%
    val: 1%
    """

    train_txt_path = os.path.join(save_root, "train", "labels")
    train_images_path = os.path.join(save_root, "train", "images")
    val_txt_path = os.path.join(save_root, "val", "labels")
    val_images_path = os.path.join(save_root, "val", "images")

    os.makedirs(train_txt_path, exist_ok=True)
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(val_txt_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)

    image_dict = build_image_dict(data)
    annotation_dict = build_annotation_dict(data)

    # 建立 category_id -> class_id 的对应关系
    category_dict = {}

    # 建立 class_name -> class_id 的对应关系
    classes_dict = {}

    for index, category in enumerate(data["categories"]):
        category_id = str(category["id"])
        class_name = category["name"]

        category_dict[category_id] = index
        classes_dict[class_name] = index

    # 创建data文件夹
    data_path = "./data"
    os.makedirs(data_path, exist_ok=True)

    classes_json_path = os.path.join(data_path, "cowboyoutfits_classes.json")
    with open(classes_json_path, "w") as f:
        json.dump(classes_dict, f, ensure_ascii=False, indent=4)

    names_path = os.path.join(data_path, "cowboyoutfits_label.names")
    with open(names_path, "w") as f:
        for class_name in classes_dict:
            f.write(class_name + "\n")

    image_items = list(image_dict.items())
    random.shuffle(image_items)
    val_num = int(len(image_items) * val_ratio)

    val_items = image_items[:val_num]
    train_items = image_items[val_num:]

    print("total images: {}".format(len(image_items)))
    print("train images: {}".format(len(train_items)))
    print("val images: {}".format(len(val_items)))


    def save_image_and_label(image_id, image, save_images_path, save_txt_path):

        img_name = image["file_name"]
        img_path = os.path.join(images_dir, img_name)

        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)
        img_width = image["width"]
        img_height = image["height"]

        annotations = annotation_dict[image_id]
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(save_txt_path, label_name)

        with open(label_path, "w") as f:
            for annotation in annotations:
                category_id = str(annotation["category_id"])
                class_index = category_dict[category_id]
                # COCO格式：
                # [左上角x, 左上角y, box宽, box高]
                x, y, w, h = annotation["bbox"]
                if w <= 0 or h <= 0:
                    print("Warning: in '{}' there are some bbox w/h <= 0".format(img_name))
                    continue

                # 转换成YOLO格式
                # [中心x, 中心y, box宽, box高]
                xc = x + w / 2
                yc = y + h / 2
                # 转换成相对坐标
                xc = round(xc / img_width, 6)
                yc = round(yc / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(class_index), str(xc), str(yc), str(w), str(h)]
                f.write(" ".join(info) + "\n")

        path_copy_to = os.path.join(save_images_path, img_name)
        if os.path.exists(path_copy_to) is False:
            shutil.copyfile(img_path, path_copy_to)

    for image_id, image in tqdm(train_items,desc="translate train file..."):
        save_image_and_label(image_id, image, train_images_path, train_txt_path)

    for image_id, image in tqdm(val_items, desc="translate val file..."):
        save_image_and_label(image_id, image, val_images_path, val_txt_path)


def translate_test_csv(csv_path, save_root):
    """
    根据valid.csv或者test.csv，
    将图片复制到YOLO对应目录
    """

    save_images_path = os.path.join(save_root, "test", "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path, exist_ok=True)

    # 读取csv文件
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        file_names = []
        for row in reader:
            file_names.append(row["file_name"])

    # 复制图片
    for file_name in tqdm(file_names, desc="copy test file..."):
        img_path = os.path.join(images_dir, file_name)
        # 检查图片是否存在
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)
        path_copy_to = os.path.join(save_images_path, file_name)

        if os.path.exists(path_copy_to) is False:
            shutil.copyfile(img_path, path_copy_to)


def main():

    with open(train_json_path, "r") as f:
        data = json.load(f)
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"train.json lack key: {key}")
    translate_info(data, save_file_root)
    translate_test_csv(test_csv_path,save_file_root)

    print("Dataset conversion finished!")


if __name__ == "__main__":
    main()