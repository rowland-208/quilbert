import argparse
import logging
import os

from quilbert.quilbert import VoiceAssistant

def validate_access():
    if os.environ.get('PICOVOICE_ACCESS_KEY') is None:
        raise ValueError("PICOVOICE_ACCESS_KEY environment variable not set")
    if os.environ.get('OPENAI_API_KEY') is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    VoiceAssistant()

if __name__ ==  "__main__":
    main()
