"""Quilbert is a GPT powered AI voice assistant built using Python.
This project utilizes a state machine design pattern
to efficiently handle user input, process it,
and provide relevant responses.
With Quilbert, you can easily access information,
ask questions, and receive helpful responses using only your voice.

This file contains the VoiceAssistant class.
Running
>>> VoiceAssistant()
will start the state machine in the sleep state.
This state listens for the wake word "porcupine" and transitions to the listening state.
"""
import logging
import os
import string
import struct

import numpy as np
import openai
import pvcobra
import pvporcupine
import pyaudio
import pyttsx3
import statemachine
import whisper

SYSTEM_MESSAGE = """You are a helpful AI voice assistant.
Keep your responses brief, one to three sentences is best, but more is ok if needed.
Your name is Quilbert if anyone asks."""

# STOP_WORDS are all lower case with no punctuation
STOP_WORDS = [
    "stop", "halt", "cease", "desist", "pause", "discontinue", "terminate",
    "end", "freeze", "quit", "arrest", "block", "check", "curb", "interrupt",
    "nullify", "suspend", "break", "conclude", "finish", "hold", "bye", "goodbye"
]

# Only picovoice porcupine model defaults are supported
WAKE_WORDS = ["porcupine"]

WHISPER_OPTIONS = whisper.DecodingOptions(fp16=False, language="en")

punction_removal_translator = str.maketrans("", "", string.punctuation)

class VoiceAssistant(statemachine.StateMachine):
    "A friendly voice assistant"
    sleeping = statemachine.State(initial=True)
    listening = statemachine.State()
    processing = statemachine.State()

    sleep = listening.to(sleeping) | processing.to(sleeping)
    listen = sleeping.to(listening) | processing.to(listening)
    process = listening.to(processing)

    def __init__(self):
        # Initialize the voice engine
        self.tts_engine = pyttsx3.init()

        # Initialize the speech recognition model
        self.stt_model = whisper.load_model("small")

        # Initialize the wake word detector
        self.ww_handle = pvporcupine.create(
            access_key=os.environ.get('PICOVOICE_ACCESS_KEY'), keywords=WAKE_WORDS
        )

        # Initialize the audio service
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.open_audio_stream()

        self.buffer = None

        self.messages = [{"role": "system", "content": SYSTEM_MESSAGE},]

        super().__init__(self)

    def open_audio_stream(self):
        """Open a stream to the default audio input device."""
        self.stream = self.audio.open(
            rate=self.ww_handle.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.ww_handle.frame_length,
        )

    def close_audio_stream(self):
        """Close the stream to the default audio input device."""
        self.stream.stop_stream()
        self.stream.close()

    def get_audio_buffer(self):
        """Read a buffer of audio data from the default audio input device."""
        audio_bytes = self.stream.read(self.ww_handle.frame_length)
        data = struct.unpack_from("h" * self.ww_handle.frame_length, audio_bytes)
        return data

    def get_signal(self):
        """Convert buffer of ints to floats between -1 and 1."""
        data = np.array([b for data in self.buffer if data for b in data], dtype=np.float32)
        data = (data - data.min())/(data.max() - data.min())
        data = data*2 - 1
        return data

    def on_enter_sleeping(self):
        """Listen for the wake word and transition to the listening state."""
        logging.debug("sleeping")
        while True:
            data = self.get_audio_buffer()
            keyword_index = self.ww_handle.process(data)
            if keyword_index >= 0:
                self.listen()
                return

    def on_enter_listening(self):
        """Listen for user input and transition to the processing state."""
        logging.debug("listening")
        self.buffer = [None for _ in range(1024)]
        vad_handle = pvcobra.create(access_key=os.environ.get('PICOVOICE_ACCESS_KEY'))
        count_active = 0
        count_inactive = 0
        for idx,_ in enumerate(self.buffer):
            data = self.get_audio_buffer()
            self.buffer[idx] = data
            voice_probability = vad_handle.process(data)
            if voice_probability>0.1:
                count_active+=1
                count_inactive=0
            else:
                count_inactive+=1

            if count_active>10 and count_inactive>30:
                self.process()
                return
        self.sleep()

    def on_enter_processing(self):
        """Process user input and transition to the listening state."""
        logging.debug("processing")
        self.close_audio_stream()

        data = whisper.pad_or_trim(self.get_signal())
        mel = whisper.log_mel_spectrogram(data).to(self.stt_model.device)
        result = whisper.decode(self.stt_model, mel, WHISPER_OPTIONS)
        logging.debug(result.text)

        if result.text.lower().translate(punction_removal_translator) in STOP_WORDS:
            self.sleep()
            return

        self.messages.append({"role": "user", "content": result.text})
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.messages.copy()
        )["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": response})

        logging.debug(response)
        self.tts_engine.say(response)
        self.tts_engine.runAndWait()

        self.listen()

    def on_exit_processing(self):
        """Reopen stream to the default audio input device after processing."""
        self.open_audio_stream()
