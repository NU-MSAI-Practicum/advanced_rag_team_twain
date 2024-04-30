import os
import yaml

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

with open(config_path, 'r') as file:
    yaml_content = yaml.safe_load(file)


