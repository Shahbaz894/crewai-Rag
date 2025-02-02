import logging
import os

# Define log directory and file
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Ensure log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

# Create logger
logger = logging.getLogger(__name__)

# Logging functions
def log_info(message):
    """Logs an INFO message."""
    logger.info(message)

def log_error(message):
    """Logs an ERROR message."""
    logger.error(message)

def log_debug(message):
    """Logs a DEBUG message (for detailed diagnostics)."""
    logger.debug(message)

def log_exception(exception):
    """Logs an EXCEPTION (use inside except blocks)."""
    logger.exception(exception)
