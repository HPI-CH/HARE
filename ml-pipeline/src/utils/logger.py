from termcolor import colored

from utils.config import Config


def log(m: str):
    if Config.log_level != 'error':
        print(colored(m, 'cyan'))


def error(m: str):
    """
    Non blocking error log
    """
    print(colored(m, 'red'))


def verbose(m: str, end=None):
    if Config.log_level == 'verbose':
        print(colored(m, 'grey'), end=end)
