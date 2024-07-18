import csv
import matplotlib.pyplot as plt

def histograms():
    csv_path = './data/data.csv'
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]

    feature_list = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    ]
    feat_type = ["Mean", "Standard error", "Largest"]

    fig = plt.figure(figsize=(16, 9))

    malignant = "M"
    benign = "B"
    feature_start_pos = 2
    feat_type_idx = 0
    for i in range(len(feature_list) * len(feat_type)):
        if i % len(feat_type) != feat_type_idx:
            continue
        feat_idx = int(i / len(feat_type))
        col_num = i + feature_start_pos
        malignant_list = []
        benign_list = []
        for data in input_data_list:
            value = float(data[col_num])
            if data[1] == malignant:
                malignant_list.append(value)
            elif data[1] == benign:
                benign_list.append(value)
            else:
                raise RuntimeError("Wrong label")

        graph = fig.add_subplot(2, 5, feat_idx + 1)
        graph.hist(malignant_list, alpha=0.4, label="malignant", color="red")
        graph.hist(benign_list, alpha=0.4, label="benign", color="blue")
        graph.legend(loc="upper right", fontsize="7")
        graph.set_title(feature_list[feat_idx], fontsize=10)

    plt.show()