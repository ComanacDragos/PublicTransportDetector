import json
import pickle


def open_json(file):
    with open(file) as f:
        return json.load(f)


def to_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def open_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
