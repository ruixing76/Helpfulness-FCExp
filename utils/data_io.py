import json
import jsonlines


def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, file_path, is_friendly_format=True, is_verbose=False):
    if is_friendly_format:
        indent = 4
    else:
        indent = None
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)
    if is_verbose:
        print(f"Data is saved to {file_path}")


def read_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    return data


def write_file(data, file_path):
    with open(file_path, 'w') as f:
        f.write(data)


def read_jsonl(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        data = [d for d in reader]
    return data


def write_jsonl(file_path, data):
    with jsonlines.open(file_path, 'w') as writer:
        for d in data:
            writer.write(d)
    print(f"Data is saved to {file_path}")
