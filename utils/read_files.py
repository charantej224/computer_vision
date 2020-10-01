import json


def read_json_value(file_name):
    with open(file_name, 'r') as f:
        json_dict = json.load(f)
        f.close()
        return json_dict
