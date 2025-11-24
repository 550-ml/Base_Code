import logging

from logger.logger import setup_logging

setup_logging("test", log_config="logger/logger_config.json")
logger = logging.getLogger(__name__)
logger.error("This is a customer debug message")
