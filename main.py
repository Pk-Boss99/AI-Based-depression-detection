# File: main.py

import sys
import os

# Add the project root to the sys.path
# This allows importing modules from the project directory
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from utils import setup_logging
from config import GUI_STREAMLIT

# Setup logging as the very first step
setup_logging()
import logging
logger = logging.getLogger(__name__)

def main():
    """
    Main function to set up the application and launch the GUI.
    """
    logger.info("Starting the Multimodal Depression Detection System.")

    # You could add argument parsing here if needed (e.g., --config, --mode)
    # import argparse
    # parser = argparse.ArgumentParser(description="Multimodal Depression Detection System")
    # # Add arguments here
    # args = parser.parse_args()

    # Launch the GUI based on the configuration
    if GUI_STREAMLIT:
        logger.info("Launching Streamlit GUI.")
        # To run a Streamlit app programmatically, you typically use
        # streamlit.cli.main(). However, this is not recommended for
        # embedding and might have issues. The standard way is to run
        # `streamlit run gui/app.py` from the terminal.
        # For this main.py to work as a single entry point, we can
        # try to execute the streamlit command using subprocess or os.system,
        # but the cleanest way is to instruct the user to run the streamlit command.
        # Alternatively, we can import and run the app function if designed for it.

        # Option 1: Instruct user (recommended for standard Streamlit)
        print("\n" + "="*50)
        print("To run the Streamlit GUI, please open your terminal and run:")
        print(f"streamlit run {os.path.join(project_root, 'gui', 'app.py')}")
        print("="*50 + "\n")
        logger.info("Instructed user to run Streamlit command.")

        # Option 2: Attempt to run programmatically (less reliable)
        # try:
        #     from streamlit.cli import main as streamlit_main
        #     # Streamlit expects sys.argv to be set as if run from command line
        #     sys.argv = ["streamlit", "run", os.path.join(project_root, 'gui', 'app.py')]
        #     streamlit_main()
        # except ImportError:
        #     logger.error("Streamlit is not installed or cannot be imported.")
        #     print("Error: Streamlit is not installed. Please install it (`pip install streamlit`).")
        # except Exception as e:
        #     logger.error(f"Error running Streamlit app programmatically: {e}", exc_info=True)
        #     print(f"Error running Streamlit app: {e}")


    else: # Assuming Gradio
        logger.info("Launching Gradio GUI.")
        try:
            # Assuming gui/app.py contains a Gradio interface object named 'iface'
            from gui.app import iface
            iface.launch()
        except ImportError:
            logger.error("Gradio is not installed or gui/app.py does not contain a Gradio interface.")
            print("Error: Gradio is not installed or the Gradio interface is not correctly defined in gui/app.py.")
        except Exception as e:
            logger.error(f"Error launching Gradio app: {e}", exc_info=True)
            print(f"Error launching Gradio app: {e}")


    logger.info("Application finished.")

if __name__ == "__main__":
    main()
