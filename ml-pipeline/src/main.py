import argparse
import importlib

from utils.logger import log


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load-module', type=str, help="Pass module to load")

    args = parser.parse_args()
    if args.load_module:
        log(f'Loading module "{args.load_module}"')
        module_name = f"scripts.{args.load_module}"
        mod = importlib.import_module(module_name)
        mod.run()
    else:
        log(f'Loading module "train_model"')
        module_name = f"scripts.train_model"
        mod = importlib.import_module(module_name)
        mod.run()


if __name__ == '__main__':
    main()
