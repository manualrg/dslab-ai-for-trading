import yaml

def read_conf():
    with open("../../conf.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    return cfg
