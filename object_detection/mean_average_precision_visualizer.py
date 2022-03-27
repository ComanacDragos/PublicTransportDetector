import os
import re

import matplotlib.pyplot as plt


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
                if convert(val) not in file_filter[key]:
                    ok = False
                    break
            if ok:
                files[file] = read_file(f'mean_average_precisions/{directory}/{file}', model_filter, column_filter)
    return files


if __name__ == '__main__':
    out = read_files(
        {
            'iou_tp': [0.5],
            'nms_iou': [0.3],
            'score': [0.5, 0.6]
        },
        ['model_v27', 'model_v28'],
        ['model_name', 'mAP', 'Bus']
    )
    for k, v in out.items():
        print(f"{k} -> {v}")

