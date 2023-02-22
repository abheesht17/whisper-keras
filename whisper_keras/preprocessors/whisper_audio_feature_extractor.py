import librosa
import tensorflow as tf
from tensorflow import keras


class WhisperAudioFeatureExtractor(keras.layers.Layer):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        chunk_length=30,
        **kwargs,
    ):
        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.float32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_floating:
                raise ValueError(
                    f"dtype must be a floating type. Received: dtype={dtype}"
                )

        super().__init__(**kwargs)

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.chunk_length = chunk_length
        self.n_samples = self.sample_rate * self.chunk_length

        # After transposition, `self.mel_filters`'s shape is
        # `(n_fft // 2 + 1, n_mels).`
        self.mel_filters = tf.constant(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
            ),
            dtype=self.dtype,
        )
        self.mel_filters = tf.transpose(self.mel_filters)

    def extract_audio_features(self, audio):
        # Use "reflection" padding - `tf.signal.stft` uses symmetric padding
        # internally.
        audio = tf.pad(
            audio,
            paddings=[[0, 0], [self.n_fft // 2, self.n_fft // 2]],
            mode="REFLECT",
        )

        # Compute the mel spectrogram.
        stft = tf.signal.stft(
            audio,
            frame_length=self.n_fft,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
        )
        magnitudes = tf.square(tf.abs(stft[:, :-1, :]))

        mel_spec = tf.matmul(
            magnitudes,
            self.mel_filters,
        )

        def tf_log10(x):
            """
            Computes log base 10 of input tensor.

            TensorFlow does not have a native implementation of log base 10, but
            does have a log base `e` (`tf.math.log`). Hence, this short
            workaround.
            """
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        # Clamp the values to a minimum value of 1e-10. This is done to avoid
        # taking the log of 0, i.e., for numerical stability.
        mel_spec = tf.maximum(mel_spec, 1e-10)

        # Calculate the log mel spectrogram.
        log_spec = tf_log10(mel_spec)
        # Dynamic range compression.
        max_value_minus_eight = tf.math.subtract(
            tf.math.reduce_max(log_spec), tf.cast(8, dtype=log_spec.dtype)
        )
        log_spec = tf.maximum(log_spec, max_value_minus_eight)
        # Normalization.
        type_cast_four = tf.cast(4, dtype=log_spec.dtype)
        log_spec = tf.math.divide(
            tf.math.add(log_spec, type_cast_four),
            type_cast_four,
        )

        return log_spec

    def call(self, audio):
        if not isinstance(audio, (tf.Tensor, tf.RaggedTensor)):
            audio = tf.convert_to_tensor(audio)

        scalar_input = audio.shape.rank == 0
        if scalar_input:
            audio = tf.expand_dims(audio, 0)

        # Convert the tensor to a Ragged Tensor.
        audio = tf.RaggedTensor.from_tensor(audio)

        # Pad audio.
        audio_shape = audio.shape.as_list()
        audio_shape[-1] = self.n_samples
        audio = audio.to_tensor(shape=audio_shape)

        # Find the log mel spectrogram.
        log_spec = self.extract_audio_features(audio)
        return log_spec
