import ffmpeg
import numpy as np
import tensorflow as tf


def load_audio(file, sample_rate=16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary.

    Note: Tried using `tensorflow-io`, but it apparently uses a different
    function for resampling. Hence, sticking to `ffmpeg-python` for now.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output(
                "-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate
            )
            .run(
                cmd=["ffmpeg", "-nostdin"],
                capture_stdout=True,
                capture_stderr=True,
            )
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    out = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    # Convert to TensorFlow tensor.
    out = tf.constant(out)
    return out
