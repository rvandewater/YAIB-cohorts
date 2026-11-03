import yaml

with open("../config.yaml") as stream:  # this path is relative to the calling environment
    conf = yaml.safe_load(stream)

# out_dir may contain {backend} to keep the outputs of the ricu and WeavEHR backends apart
conf["out_dir"] = conf["out_dir"].format(backend=conf.get("backend", "ricu"))
