import os
import argparse

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RESOURCES_FOLDER = os.path.join(PACKAGE_DIRECTORY, "resources")
QUESTION_WORDS = ["who", "what", "when", "where", "why", "how"]

def get_resources_folder():
    if not os.path.isdir(RESOURCES_FOLDER):
        os.mkdir(RESOURCES_FOLDER)
    return RESOURCES_FOLDER

class RawTextAndArgumentDefaultsHelpFormatter( argparse.RawTextHelpFormatter, 
                                               argparse.ArgumentDefaultsHelpFormatter ):
    """
    Helper class to format argparse help messages.
    Allows messages to be formatted as raw text while also
    showing default argument values.
    """
    pass
