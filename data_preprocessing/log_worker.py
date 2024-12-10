import logging
from datetime import datetime


def add_info_logging(text_info):
    current_time = datetime.now()
    str_time = current_time.strftime("%d:%H:%M")
    logging.info(f"time:  {str_time} {text_info}")
