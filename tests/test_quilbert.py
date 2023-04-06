"""Test the VoiceAssistant class:
1. Load recorded audio from a bytes file.
2. Mock pyaudio to return the audio bytes.
3. Mock openai to return a test response.
4. Mock pyttsx3 to do nothing.
5. Run the VoiceAssistant class.
Expected behavior:
The bytes contain voice data that says:
"Porcupine <pause> What's the capital of Europe?"
The assistant should:
1. Recognize the wake word porcupine.
2. Start listening for activity.
3. Decode the audio for "What's the capital of Europe?"
4. Send the message to OpenAI.
5. Receive a response from OpenAI.
6. Speak the response.
"""
from unittest.mock import Mock

import pytest

from quilbert.quilbert import VoiceAssistant

def test_quilbert(monkeypatch):
    """Test the VoiceAssistant class:
    1. Load recorded audio from a bytes file.
    2. Mock pyaudio to return the audio bytes.
    3. Mock openai to return a test response.
    4. Mock pyttsx3 to do nothing.
    5. Run the VoiceAssistant class.
    Expected behavior:
    The bytes contain voice data that says:
    "Porcupine <pause> What's the capital of Europe?"
    The assistant should:
    1. Recognize the wake word porcupine.
    2. Start listening for activity.
    3. Decode the audio for "What's the capital of Europe?"
    4. Send the message to OpenAI.
    5. Receive a response from OpenAI.
    6. Speak the response.
    """
    # mock pyaudio
    mock_pyaudio = Mock()
    mock_pyaudio_pyaudio = Mock()
    mock_pyaudio_stream = Mock()
    mock_pyaudio.PyAudio.return_value = mock_pyaudio_pyaudio
    def next_stream():
        yield mock_pyaudio_stream
        raise RuntimeError("test")
    mock_pyaudio_pyaudio.open = Mock(side_effect=next_stream())
    with open("tests/audio.pcm", "rb") as file_handle:
        audio_bytes = file_handle.read()
    def next_bytes():
        while True:
            for offset in range(0, len(audio_bytes), 1024):
                yield audio_bytes[offset:offset+1024]
    mock_pyaudio_stream.read = Mock(side_effect=next_bytes())
    mock_pyaudio_stream.stop_stream.return_value = None
    mock_pyaudio_stream.close.return_value = None
    monkeypatch.setattr("quilbert.quilbert.pyaudio", mock_pyaudio)

    # mock openai
    mock_openai = Mock()
    mock_openai.ChatCompletion.create.return_value = {
        "choices": [{"message": {"content": "test"}}]
    }
    monkeypatch.setattr("quilbert.quilbert.openai", mock_openai)

    # mock pyttsx3
    mock_pyttsx3 = Mock()
    mock_pyttsx3_engine = Mock()
    mock_pyttsx3.init.return_value = mock_pyttsx3_engine
    mock_pyttsx3_engine.say.return_value = None
    mock_pyttsx3_engine.runAndWait.return_value = None
    monkeypatch.setattr("quilbert.quilbert.pyttsx3", mock_pyttsx3)

    with pytest.raises(RuntimeError, match="test"):
        VoiceAssistant()

    assert mock_openai.ChatCompletion.create.call_count == 1
    assert mock_openai.ChatCompletion.create.call_args.kwargs["messages"][0]["role"] == "system"
    assert mock_openai.ChatCompletion.create.call_args.kwargs["messages"][-1] == {
        "role": "user", "content": "What's the capital of Europe?"
    }

    assert mock_pyttsx3_engine.say.call_count == 1
    assert mock_pyttsx3_engine.runAndWait.call_count == 1
    assert mock_pyttsx3_engine.say.call_args.args == ("test",)
