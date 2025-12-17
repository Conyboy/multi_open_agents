#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
logger file
"""

import logging
import os
from enum import Enum
from logging.handlers import RotatingFileHandler

LOG_LEVEL = logging.INFO
DEFAULT_LOGGER = "dky-tslib"
DEFAULT_LOGGER_FILE_NAME = "output"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class Logger(object):
    """
    log class

    Args:
        name(str, optional): module name

    Attributes:
        logger(Logger) : a logger with the specified name which created  by python logging
        lvl(Enum): log level

    """
    level = Enum('level', {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    })
    logger = None
    lvl = None

    def __init__(self, name=DEFAULT_LOGGER, log_name=DEFAULT_LOGGER_FILE_NAME):
        # Do basic configuration(level and format) for the logging system
        logging.basicConfig(level=LOG_LEVEL, format=DEFAULT_LOG_FORMAT)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVEL)
        # create file handler which logs messages
        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.dirname(current_path)
        if not os.path.exists(f'{root_path}/logs'):
            os.mkdir(f'{root_path}/logs')
        fh = RotatingFileHandler(f'{root_path}/logs/{log_name}.log', maxBytes=10 * 1024 * 1024, backupCount=5)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def __getattr__(self, name):
        """
        Args:
            name(str): log level name

        Returns
            Logger: returned logger

        Raises:
            AttributeError: An error occurred where the Attr not Correct

        """
        if name in ('debug', 'info', 'warning', 'error', 'critical'):
            self.lvl = self.level[name].value
            return self
        else:
            raise AttributeError('Attr not Correct')

    def __call__(self, msg):
        """
        Args:
            msg(str):  message to be printed

        Returns
            None

        """
        self.logger.log(self.lvl, msg)
