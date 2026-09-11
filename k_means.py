import json
import numpy as np


def iou(box, clusters):
    """
    计算一个真实框和9个聚类中心之间的IoU
    box: [w, h]
    clusters: [9, 2]
    """
    # 逐个取较小值
    w = np.minimum(clusters[:, 0], box[0])
    h = np.minimum(clusters[:, 1], box[1])

    # 与各锚框的最大重合面积
    intersection = w * h

    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    union = box_area + cluster_area - intersection

    return intersection / union  # 最大iou


def kmeans(boxes, k=9, max_iter=1000):
    """
    使用IoU作为距离进行K-means聚类
    """

    # 随机选择9个真实框作为初始聚类中心
    clusters = boxes[np.random.choice(len(boxes), k, replace=False)]
    for _ in range(max_iter):
        # 计算每个真实框与9个anchor之间的距离
        distances = np.array([
            1 - iou(box, clusters)
            for box in boxes
        ])

        # 每个真实框选择距离最近的anchor[len(boxes)]
        nearest = np.argmin(distances, axis=1)
        old_clusters = clusters.copy()

        # 更新9个anchor
        for i in range(k):
            if np.sum(nearest == i) > 0:
                clusters[i] = np.mean(boxes[nearest == i], axis=0)
        # 如果anchor基本不再变化，就结束
        if np.allclose(old_clusters, clusters):
            break

    return clusters


if __name__ == "__main__":
    json_path = "./cowboyoutfits/train.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    train_image_ids = set()
    for image in data["images"]:
        train_image_ids.add(image["id"])

    boxes = []
    for ann in data["annotations"]:
        if ann["image_id"] not in train_image_ids:
            continue

        x, y, w, h = ann["bbox"]
        boxes.append([w, h])
    boxes = np.array(boxes)

    print("训练集图片数量:", len(train_image_ids))
    print("训练集真实bbox数量:", len(boxes))

    anchors = kmeans(boxes,k=9)
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    print("\n9个Anchor：")
    print("anchors = " + ", ".join("{:.0f},{:.0f}".format(w, h)for w, h in anchors))
