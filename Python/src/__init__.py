import yaml

with open("../config.yaml") as stream:  # this path is relative to the calling environment
    conf = yaml.safe_load(stream)