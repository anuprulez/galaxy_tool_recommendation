import os
import numpy as np
import json
import h5py


def read_file(file_path):
    """
    Read a file
    """
    with open(file_path, "r") as json_file:
        file_content = json.loads(json_file.read())
    return file_content


def write_file(file_path, content):
    """
    Write a file
    """
    remove_file(file_path)
    with open(file_path, "w") as json_file:
        json_file.write(json.dumps(content))


def save_processed_workflows(file_path, unique_paths):
    workflow_paths_unique = ""
    for path in unique_paths:
        workflow_paths_unique += path + "\n"
    with open(file_path, "w") as workflows_file:
        workflows_file.write(workflow_paths_unique)


def format_tool_id(tool_link):
    """
    Extract tool id from tool link
    """
    tool_id_split = tool_link.split("/")
    tool_id = tool_id_split[-2] if len(tool_id_split) > 1 else tool_link
    return tool_id


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def save_model(data, model_path):
    # save files
    """
    Create an h5 file with the trained weights and associated dicts
    """
    hf_file = h5py.File(model_path, 'w')
    for key in data:
        value = data[key]
        if key in hf_file:
            hf_file.modify(key, json.dumps(value))
        else:
            hf_file.create_dataset(key, data=json.dumps(value))
    hf_file.close()
    

def load_model(model_path):
    model = h5py.File(model_path, 'r')
    dictionary = json.loads(model.get('data_dictionary').value)
    paths = json.loads(model.get('multilabels_paths').value)
    c_tools = json.loads(model.get('compatible_next_tools').value)
    class_weights = json.loads(model.get('class_weights').value)
    return paths, dictionary, c_tools, class_weights
