import logging
import os
from config import LOG_FILE, LOG_LEVEL

def setup_logging():
    """
    Sets up the logging configuration for the application.
    Logs will be written to the file specified in config.py.
    """
    log_directory = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler() # Also log to console
        ]
    )
    logging.info("Logging configured.")

# Example usage (optional, can be removed or kept for testing)
if __name__ == "__main__":
    setup_logging()
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
