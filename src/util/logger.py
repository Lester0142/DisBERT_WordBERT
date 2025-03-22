import os
import logging

from time import strftime

def create_log_dir(log_dir):
    """
    Generate a directory path for logging.
    
    Creates a new directory with an incremental index and timestamp inside the given log_dir.
    If log_dir does not exist, it is created.

    :param log_dir: Path to the main logging directory.
    :return: Full path to the newly created log directory.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_dirs = [d for d in os.listdir(log_dir) if d.split("_")[0].isdigit()]
    idx = max([int(d.split("_")[0]) for d in log_dirs], default=-1) + 1
    
    cur_log_dir = os.path.join(log_dir, f"{idx}_{strftime('%Y%m%d-%H%M')}")
    os.makedirs(cur_log_dir)
    
    return cur_log_dir

class Logger:
    """
    A logger class to handle logging messages to both console and a log file.
    """
    def __init__(self, log_dir, log_file="training.log", log_level=logging.INFO):
        """
        Initializes the Logger.

        :param log_dir: Directory where log files will be stored.
        :param log_file: Name of the log file. Defaults to "training.log".
        :param log_level: Logging level. Defaults to logging.INFO.
        """
        self.log_dir = log_dir
        self.logger = logging.getLogger(log_dir)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        log_file_path = os.path.join(log_dir, log_file)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("[%(lineno)d]%(asctime)s: %(message)s"))

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def log(self, level, msg):
        """
        Logs a message with the specified log level.

        :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
        :param msg: Message to log.
        """
        self.logger.log(level, msg)

    def info(self, msg):
        """
        Logs an informational message.

        :param msg: Message to log.
        """
        self.log(logging.INFO, msg)
    
    def debug(self, msg):
        """
        Logs a debug message.

        :param msg: Message to log.
        """
        self.log(logging.DEBUG, msg)
    
    def warning(self, msg):
        """
        Logs a warning message.

        :param msg: Message to log.
        """
        self.log(logging.WARNING, msg)
    
    def error(self, msg):
        """
        Logs an error message.

        :param msg: Message to log.
        """
        self.log(logging.ERROR, msg)
    
    def close(self):
        """
        Closes the logger by removing all handlers and shutting down logging.
        """
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        logging.shutdown()
