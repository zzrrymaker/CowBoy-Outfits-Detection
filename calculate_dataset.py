"""
该脚本有3个功能：
1.统计训练集和验证集的数据并生成相应.txt文件
2.创建.data文件，记录classes个数, train,val,test数据集文件(.txt)路径和label.names文件路径
3.根据yolov3-spp.cfg创建my_yolov3.cfg文件修改其中的predictor filters, yolo classes以及anchors参数
"""
import os

train_annotation_dir = "./cowboyoutfits_yolo_dataset/train/labels/"
val_images_dir = "./cowboyoutfits_yolo_dataset/val/images/"
test_images_dir = "./cowboyoutfits_yolo_dataset/test/images/"
classes_label = "./data/cowboyoutfits_label.names"
cfg_path = "./cfg/yolov3-spp.cfg"

assert os.path.exists(train_annotation_dir), "train_annotation_dir not exist!"
assert os.path.exists(classes_label), "classes_label not exist!"
assert os.path.exists(cfg_path), "cfg_path not exist!"

def calculate_trian_data_txt(txt_path, annotation_dir):
    # create my_data.txt file that record image list
    with open(txt_path, "w") as w:
        for file_name in os.listdir(annotation_dir):
            if file_name == "classes.txt":
                continue

            img_path = os.path.join(annotation_dir.replace("labels", "images"),
                                    file_name.split(".")[0]) + ".jpg"
            line = img_path + "\n"
            assert os.path.exists(img_path), "file:{} not exist!".format(img_path)
            w.write(line)

def calculate_data_txt(txt_path, images_dir):
    with open(txt_path, "w") as w:
        for file_name in os.listdir(images_dir):
            img_path = os.path.join(images_dir, file_name)
            line = img_path + "\n"
            assert os.path.exists(img_path), "file:{} not exist!".format(img_path)
            w.write(line)

def create_data_data(create_data_path, train_path, val_path, test_path, classes_info):
    # create my_data.data file that record classes, train, valid, test and names info.
    with open(create_data_path, "w") as w:
        w.write("classes={}".format(len(classes_info)) + "\n")  # 记录类别个数
        w.write("train={}".format(train_path) + "\n")           # 记录训练集对应txt文件路径
        w.write("valid={}".format(val_path) + "\n")             # 记录验证集对应txt文件路径
        w.write("test={}".format(test_path) + "\n")             # 记录测试集对应txt文件路径
        w.write("names=data/cowboyoutfits_label.names" + "\n")        # 记录label.names文件路径


def change_and_create_cfg_file(classes_info, save_cfg_path="./cfg/my_yolov3.cfg"):
    # create my_yolov3.cfg file changed predictor filters and yolo classes param.
    # this operation only deal with yolov3-spp.cfg
    filters_lines = [636, 722, 809]  # predictor输出通道数
    classes_lines = [643, 729, 816]  # yolo层输出类别数
    anchors_lines = [642, 728, 815]  # # yolo层预设锚框大小

    anchors =[[43,21], [103,60], [215,97],
              [122,186], [323,182], [222,334],
              [560,328], [369,499], [686,637]]
    # 只读模式不改变原yolov3-spp.cfg文件
    cfg_lines = open(cfg_path, "r").readlines()

    for i in filters_lines:
        assert "filters" in cfg_lines[i-1], "filters param is not in line:{}".format(i-1)
        output_num = (5 + len(classes_info)) * 3
        cfg_lines[i-1] = "filters={}\n".format(output_num)

    for i in classes_lines:
        assert "classes" in cfg_lines[i-1], "classes param is not in line:{}".format(i-1)
        cfg_lines[i-1] = "classes={}\n".format(len(classes_info))

    for i in anchors_lines:
        assert "anchors" in cfg_lines[i - 1], "anchors param is not in line:{}".format(i - 1)
        anchors_str = ", ".join("{:.0f},{:.0f}".format(w, h) for w, h in anchors)
        cfg_lines[i - 1] = "anchors={}\n".format(anchors_str)


    with open(save_cfg_path, "w") as w:
        w.writelines(cfg_lines)


def main():
    # 统计训练集、验证集、测试集的数据并生成相应txt文件
    train_txt_path = "data/my_train_data.txt"
    val_txt_path = "data/my_val_data.txt"
    test_txt_path = "data/my_test_data.txt"
    calculate_trian_data_txt(train_txt_path, train_annotation_dir)
    calculate_data_txt(val_txt_path, val_images_dir)
    calculate_data_txt(test_txt_path, test_images_dir)


    classes_info = [line.strip() for line in open(classes_label, "r").readlines() if len(line.strip()) > 0]
    # 创建data.data文件，记录classes个数, train以及val数据集文件(.txt)路径和label.names文件路径
    create_data_data("./data/my_data.data", train_txt_path, val_txt_path, test_txt_path, classes_info)
    # 根据yolov3-spp.cfg创建my_yolov3.cfg文件修改其中的predictor filters以及yolo classes参数(这两个参数是根据类别数改变的)
    change_and_create_cfg_file(classes_info)


if __name__ == '__main__':
    main()
