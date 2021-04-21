import logging


def log_config(filename):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    std_formatter = logging.Formatter('%(message)s')
    std_handler = logging.StreamHandler()
    std_handler.setLevel(logging.DEBUG)
    std_handler.setFormatter(std_formatter)

    f_formatter = logging.Formatter('[%(asctime)s] %(message)s')
    f_handler = logging.FileHandler(filename)
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(f_formatter)

    logger.addHandler(f_handler)
    logger.addHandler(std_handler)

    return logger