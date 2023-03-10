"""Whisper decoder layer inspired from `keras_nlp.layers.TransformerDecoder`."""


from tensorflow import keras

from keras_nlp.layers.transformer_decoder import TransformerDecoder
from keras_nlp.utils.keras_utils import clone_initializer
from whisper_keras.layers.whisper_multi_head_attention import (
    WhisperMultiHeadAttention,
)


class WhisperDecoderLayer(TransformerDecoder):
    """Whisper decoder layer.

    Inherits from `keras_nlp.layers.TransformerDecoder`, and overrides the
    `_build` method to use the `whisper_keras.layers.WhisperMultiHeadAttention` layer
    instead of `keras.layers.MultiHeadAttention`.
    """

    def _build(self, input_shape, has_cross_attention):
        # Create layers based on input shape.
        self._built = True
        self._input_shape = input_shape
        self._has_cross_attention = has_cross_attention
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = input_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        head_dim = int(hidden_dim // self.num_heads)

        # Self attention layers.
        self._self_attention_layer = WhisperMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._self_attention_layer._build_from_signature(
            query=input_shape,
            value=input_shape,
        )
        self._self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Cross attention layers are optional.
        self._cross_attention_layer = None
        if has_cross_attention:
            self._cross_attention_layer = WhisperMultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=head_dim,
                value_dim=head_dim,
                dropout=self.dropout,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
            )
            self._cross_attention_layer._build_from_signature(
                query=input_shape,
                value=input_shape,
            )
            self._cross_attention_layernorm = keras.layers.LayerNormalization(
                epsilon=self.layer_norm_epsilon,
            )
            self._cross_attention_dropout = keras.layers.Dropout(
                rate=self.dropout,
            )

        # Feedforward layers.
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )
