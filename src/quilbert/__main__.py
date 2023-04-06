"""Quilbert is a GPT powered AI voice assistant built using Python.
This project utilizes a state machine design pattern
to efficiently handle user input, process it,
and provide relevant responses.
With Quilbert, you can easily access information,
ask questions, and receive helpful responses using only your voice.

This file defines the command line interface for Quilbert.
To run Quilbert, run the following command:
>>> python -m quilbert

If you installed Quilbert using pip, you can run the following shell command:
$ quilbert
"""
import argparse
import logging
import os

from quilbert.quilbert import VoiceAssistant

def validate_access():
    """Validate that the PICOVOICE_ACCESS_KEY and OPENAI_API_KEY environment variables are set."""
    if os.environ.get('PICOVOICE_ACCESS_KEY') is None:
        raise ValueError("PICOVOICE_ACCESS_KEY environment variable not set")
    if os.environ.get('OPENAI_API_KEY') is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")

def main():
    """Parse command line arguments and start the voice assistant."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    VoiceAssistant()

if __name__ ==  "__main__":
    main()
