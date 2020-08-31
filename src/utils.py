import yaml

def read_conf():
    with open("../../conf.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)
    return cfg
