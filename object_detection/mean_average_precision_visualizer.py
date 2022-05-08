import os
import re

import matplotlib.pyplot as plt
import numpy as np


def convert(string):
    """
    If the string represents a float, return the float value of the string, otherwise
    return the string
    """
    try:
        return float(string)
    except ValueError:
        return string


def read_file(file, model_filter, column_filter):
    with open(file) as f:
        lines = f.readlines()
        index_to_attr = {i: attr for i, attr in enumerate(lines[0].split(','))
                         if attr in column_filter}
        output = {attr: [] for attr in index_to_attr.values()}
        for line in lines[1:]:
            values = line.split(',')
            if values[0] in model_filter:
                for i, val in enumerate(values):
                    if i in index_to_attr:
                        output[index_to_attr[i]].append(convert(val))
        return output


def extract_data_from_file(file):
    data = {}
    for pair in re.split('(?<=[0-9])_', file[:file.rfind(".")]):
        split = pair.split('=')
        data[split[0]] = convert(split[1])
    return data


def read_files(file_filter, model_filter, column_filter):
    """
    filter is a dictionary of the form
    keys are the attributes that are filtered
    values are the values of the attributes
    """
    files = {}
    for directory in os.listdir('mean_average_precisions'):
        for file in os.listdir(f'mean_average_precisions/{directory}'):
            ok = True
            for key, val in extract_data_from_file(file).items():
                if convert(val) not in file_filter[key] and len(file_filter[key]) > 0:
                    ok = False
                    break
            if ok:
                files[file] = read_file(f'mean_average_precisions/{directory}/{file}', model_filter, column_filter)
    return files


def visualize(iou_tp, nms_iou, score, models, metrics):
    data = read_files(
        {
            'iou_tp': [] if iou_tp is None else [iou_tp],
            'nms_iou': [] if nms_iou is None else [nms_iou],
            'score': [] if score is None else [score]
        },
        models.keys(),
        ['model_name', *metrics]
    )
    if iou_tp is None:
        x_axis_attr = 'iou_tp'
        x_axis_label = 'True positive IOU threshold'
    elif nms_iou is None:
        x_axis_attr = 'nms_iou'
        x_axis_label = 'NMS IOU threshold'
    else:
        x_axis_attr = 'score'
        x_axis_label = 'Score threshold'
    x = []
    for file in data:
        file_data = extract_data_from_file(file)
        x.append(file_data[x_axis_attr])

    fig, axs = plt.subplots(len(metrics), figsize=(5, 20))

    for (i, metric) in enumerate(metrics):
        y = []
        legend = []
        for value in data.values():
            legend = value['model_name']
            y.append(value[metric])
        y = np.asarray(y)
        axs[i].title.set_text(metric)
        y_ticks = []
        for (j, model) in enumerate(legend):
            y_values = y[:, j]
            axs[i].plot(x, y_values, label=f"{model} {models[model]}")
            max_index = np.argmax(y_values)
            max_y_value = y_values[max_index]
            y_ticks.append(max_y_value)
            axs[i].plot(x[max_index], max_y_value, 'r*')

        filtered_y_ticks = [round(y_ticks[0], 2)]
        for tick in y_ticks[1:]:
            if tick - filtered_y_ticks[-1] > 1e-2:
                filtered_y_ticks.append(round(tick, 2))

        axs[i].set_xticks(x)
        axs[i].set_yticks(filtered_y_ticks)
        axs[i].legend()
        axs[i].set_xlabel(x_axis_label)
        if metric != 'mAP':
            axs[i].set_ylabel('AP')
        else:
            axs[i].set_ylabel('mAP')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    out = read_files(
        {
            'iou_tp': [0.5],
            'nms_iou': [],
            'score': [0.4]
        },
        ['model_v27', 'model_v28'],
        ['model_name', 'mAP', 'Bus', 'Car', 'Vehicle registration plate']
    )
    for k, v in out.items():
        print(f"{k} -> {v}")

    models = {
        #'model_v36': "dropout=0.3",
        #'model_v37': "dropout=0.4",
        #'model_v38': "dropout=0.5",
        #'model_v39': "dropout=0.2",

        #'model_v33': "bs=32",
        #'model_v35': "bs=16",
        #'model_v36': "bs=8",

        'model_v39': "cutout=64",
        'model_v40': "cutout=32",
        'model_v41': "cutout=128",
        'model_v42': "cutout=192",
    }

    visualize(0.5, 0.3, None,
              models,
              ['mAP', 'Bus', 'Car', 'Vehicle registration plate']
              )