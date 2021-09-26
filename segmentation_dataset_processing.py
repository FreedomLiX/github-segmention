import argparse
import base64
import json
import os
import os.path as osp
import shutil

import cv2
import imgviz
import PIL.Image
import numpy as np
from labelme.logger import logger
from labelme import utils


def change_label(json_file):
    # 标注错误，更改标签
    data = json.load(open(json_file))
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        print(shape)
        label_name = shape["label"]
        if label_name == "pointer":
            shape["label"] = "scale"
        else:
            shape["label"] = "pointer"
    with open(json_file, 'w') as dump_f:
        json.dump(data, dump_f)


def json_to_label_png(json_file, out_path):
    """
    json 文件，转换成label图像 以及其他可视化图像，
    """
    json_name = json_file.split('/')[-1].split('.')[0]
    if out_path is None:
        out_dir = osp.basename(json_file).replace(".", "_")
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = out_path
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    image_path = os.path.join(out_dir, "images")
    if not osp.exists(image_path):
        os.mkdir(image_path)
    label_path = os.path.join(out_dir, "annotations")
    if not osp.exists(label_path):
        os.mkdir(label_path)

    label_rgb = os.path.join(out_dir, "label_rgb")
    if not osp.exists(label_rgb):
        os.mkdir(label_rgb)

    label_viz_path = os.path.join(out_dir, "label_viz")
    if not osp.exists(label_viz_path):
        os.mkdir(label_viz_path)

    label_text_path = os.path.join(out_dir, "label_text")
    if not osp.exists(label_text_path):
        os.mkdir(label_text_path)

    data = json.load(open(json_file))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)
    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )
    lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
    lbl_pil.save(os.path.join(label_path, json_name + ".png"))
    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
    )
    PIL.Image.fromarray(img).save(osp.join(image_path, json_name + ".png"))
    utils.lblsave(osp.join(label_rgb, json_name + "_label.png"), lbl)

    PIL.Image.fromarray(lbl_viz).save(osp.join(label_viz_path, json_name + "_label_viz.png"))
    with open(osp.join(label_text_path, json_name + "_label_names.txt"), "w") as f:
        for lbl_name in label_names:
            f.write(lbl_name + "\n")
    logger.info("Saved to: {}".format(out_dir))


def dataset_txt(dataset_path, train_val=0.8):
    """
    dataset clip
    dataset_path : 数据路径包含两个文件夹，分别是：原始图像以及标签图像文件夹
    train_val：数据切分比例
    """
    import random
    image_path = os.path.join(dataset_path, "images")
    if not osp.exists(image_path):
        raise FileExistsError
    label_path = os.path.join(dataset_path, "annotations")
    if not osp.exists(label_path):
        raise FileExistsError
    images, labels = sorted([os.path.join(image_path, file) for file in os.listdir(image_path)]), \
                     sorted([os.path.join(label_path, file) for file in os.listdir(label_path)])
    assert len(images) == len(labels)
    # dataset clip
    images_labels = list(zip(images, labels))
    random.seed(42)
    random.shuffle(images_labels)
    images[:], labels[:] = zip(*images_labels)
    i = int(len(images) * train_val)
    train_imagesPaths = images[:i]
    val_imagesPaths = images[i:]
    train_labelsPaths = labels[:i]
    val_labelsPaths = labels[i:]
    train_images_folder = os.path.join(image_path, "train")
    val_images_folder = os.path.join(image_path, "val")
    train_labels_folder = os.path.join(label_path, "train")
    val_labels_folder = os.path.join(label_path, "val")
    if not osp.exists(train_images_folder): os.mkdir(train_images_folder)
    if not osp.exists(val_images_folder): os.mkdir(val_images_folder)
    if not osp.exists(train_labels_folder): os.mkdir(train_labels_folder)
    if not osp.exists(val_labels_folder):  os.mkdir(val_labels_folder)

    [shutil.move(file, train_images_folder) for file in train_imagesPaths]
    [shutil.move(file, val_images_folder) for file in val_imagesPaths]
    [shutil.move(file, train_labels_folder) for file in train_labelsPaths]
    [shutil.move(file, val_labels_folder) for file in val_labelsPaths]

    with open(os.path.join(dataset_path, "train.txt"), "w") as f:
        for image, label in zip(sorted(train_imagesPaths), sorted(train_labelsPaths)):
            f.write("images/train/" + image.split("/")[-1] + " " + "annotations/train/" + label.split("/")[-1] + "\n")
    with open(os.path.join(dataset_path, "val.txt"), "w") as f:
        for image, label in zip(sorted(val_imagesPaths), sorted(val_labelsPaths)):
            f.write("images/val/" + image.split("/")[-1] + " " + "annotations/val/" + label.split("/")[-1] + "\n")


if __name__ == "__main__":
    # json_path = "/home/dixn/PycharmProjects/pytorch-segmentation/图像分割/JSON"
    # out = "./图像分割"
    # # json_path = "/home/dixn/PycharmProjects/pytorch-segmentation/图像分割/json_data"
    # for json_file in os.listdir(json_path):
    #     json_to_label_png(os.path.join(json_path, json_file), out)
    # print(os.path.join(json_path, json_file))
    # change_label(os.path.join(json_path, json_file))
    dataset_txt("/home/dixn/PycharmProjects/pytorch-segmentation/图像分割/dataset")
