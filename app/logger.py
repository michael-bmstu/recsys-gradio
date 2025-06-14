import logging

def setup_logger(path: str = 'log/app.log'):
    logger = logging.getLogger("RecSys")
    logger.setLevel(logging.DEBUG)

    ch = logging.FileHandler(path, mode='a', encoding='utf-8')
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logger()