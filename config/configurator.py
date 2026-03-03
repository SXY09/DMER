import os
import yaml
import argparse
from .config_loader import ConfigLoader

def update_configs(configs, up):
    """ recursively update configs """
    for k, v in up.items():
        if k not in configs:
            configs[k] = v
        elif isinstance(v, dict):
            update_configs(configs[k], v)
        else:
            configs[k] = v

def parse_configure():
    parser = argparse.ArgumentParser(description='DMER')
    parser.add_argument('--model', type=str, default="REACT_Memory", help='Model name')

    parser.add_argument('--logname', type=str, default=None, help='Log name')
    parser.add_argument('--config_list', type=str, default=None, help='Config list')
    
    parser.add_argument('--dataset', type=str, default="CDR", choices=['CDR', 'GDA', 'CHR'], help='Dataset name')

    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    args = parser.parse_args()

    if args.model == None:
        raise Exception("Please provide the model name through --model.")
    model_name = args.model.lower()
    if not os.path.exists('./config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")


    yml_fn = './config/modelconf/{}.yml'.format(model_name)
    configs = ConfigLoader().load_from(yml_fn)

    configs['model']['name'] = configs['model']['name'].lower()
    configs['model']['logname'] = configs['model']['name'] if (not args.logname) else args.logname
    if args.dataset is not None:
        configs['data']['name'] = args.dataset

    if args.config_list:
        config_list = args.config_list.split(',')
        for config_name in config_list:
            fn = f"./config/override/{config_name}.yml"
            if not os.path.exists(fn):
                raise Exception(f"Config file {fn} does not exist.")
            with open(fn, encoding='utf-8') as f:
                config_data = f.read()
                config = yaml.safe_load(config_data)
                update_configs(configs, config)

    return configs

configs = parse_configure()
