"""Whisper backbone model."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.position_embedding import PositionEmbedding
from keras_nlp.layers.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_nlp.models.backbone import Backbone
from src.layers.transformer_decoder import TransformerDecoder
from src.layers.transformer_encoder import TransformerEncoder


def bart_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras.utils.register_keras_serializable(package="keras_nlp")
class WhisperBackbone(Backbone):
    """BART encoder-decoder network.

    This class implements a Transformer-based encoder-decoder model as
    described in
    ["BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"](https://arxiv.org/abs/1910.13461).

    The default constructor gives a fully customizable, randomly initialized BART
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer encoder layers and
            transformer decoder layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        num_mels: int. The number of mel-frequency filters. For now, only 80
            works.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.

    Examples:
    ```python
    input_data = {
        "encoder_token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
        "encoder_padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "decoder_token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
        "decoder_padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], shape=(1, 12)
        ),
    }

    # Randomly initialized BART encoder-decoder model with a custom config
    model = keras_nlp.models.WhisperBackbone(
        vocabulary_size=50265,
        num_layers=6,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12,
    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        num_mels=80,
        dropout=0.0,
        max_source_sequence_length=1500,
        max_target_sequence_length=448,
        **kwargs,
    ):
        # Encoder inputs
        encoder_feature_input = keras.Input(
            shape=(None, num_mels), dtype="float32", name="encoder_features"
        )
        encoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="encoder_padding_mask"
        )

        # Decoder inputs.
        decoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )

        # ====== Encoder ======

        # === Embedding ===
        # Embed the input features. This consists of two 1D convolutional
        # layers.

        # Note: We use `padding="same"` here since that corresponds to a padding
        # size of 1.
        encoder_conv_layer_1 = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=3,
            strides=1,
            padding="same",
            name="encoder_token_embedding_conv_layer_1",
        )
        embedded_features = keras.activations.gelu(
            encoder_conv_layer_1(encoder_feature_input),
            approximate=False,
        )

        # Note: We cannot use `padding="same"` here since that corresponds to a
        # padding size of 1.5 (since stride is 2). Hence, we will manually
        # pad the input.
        embedded_features = tf.pad(
            embedded_features, paddings=[[0, 0], [1, 1], [0, 0]]
        )
        encoder_conv_layer_2 = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=3,
            strides=2,
            padding="valid",
            name="encoder_token_embedding_conv_layer_2",
        )
        embedded_features = keras.activations.gelu(
            encoder_conv_layer_2(embedded_features),
            approximate=False,
        )

        position_embedding = PositionEmbedding(
            initializer=bart_kernel_initializer(),
            sequence_length=max_source_sequence_length,
            name="encoder_position_embedding",
        )(embedded_features)

        # Sum and apply dropout to embeddings.
        x = keras.layers.Add()((embedded_features, position_embedding))
        x = keras.layers.Dropout(
            dropout,
            name="encoder_embeddings_dropout",
        )(x)

        # === Transformer Encoder Layers ===
        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=False
                ),
                layer_norm_epsilon=1e-5,
                dropout=dropout,
                kernel_initializer=bart_kernel_initializer(),
                name=f"transformer_encoder_layer_{i}",
            )(x, padding_mask=encoder_padding_mask)

        x = keras.layers.LayerNormalization(
            name="encoder_layer_norm",
            axis=-1,
            epsilon=1e-5,
            dtype=tf.float32,
        )(x)
        encoder_output = x

        # ====== Decoder ======

        # === Embedding ===
        # Embed tokens and positions.
        x = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_target_sequence_length,
            embedding_dim=hidden_dim,
            embeddings_initializer=bart_kernel_initializer(),
            name="decoder_token_and_position_embedding",
        )(decoder_token_id_input)

        # Apply dropout to embeddings.
        x = keras.layers.Dropout(
            dropout,
            name="decoder_embeddings_dropout",
        )(x)

        # === Transformer Decoder Layers ===
        # Apply successive transformer decoder blocks.
        for i in range(num_layers):
            transformer_decoder_layer = TransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=False
                ),
                layer_norm_epsilon=1e-5,
                kernel_initializer=bart_kernel_initializer(),
                name=f"transformer_decoder_layer_{i}",
                has_cross_attention=True,
            )
            x = transformer_decoder_layer(
                decoder_sequence=x,
                encoder_sequence=encoder_output,
                decoder_padding_mask=decoder_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
            )

        x = keras.layers.LayerNormalization(
            name="decoder_layer_norm",
            axis=-1,
            epsilon=1e-5,
            dtype=tf.float32,
        )(x)
        decoder_output = x

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "encoder_feature_input": encoder_feature_input,
                "encoder_padding_mask": encoder_padding_mask,
                "decoder_token_ids": decoder_token_id_input,
                "decoder_padding_mask": decoder_padding_mask,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
            },
            **kwargs,
        )

        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_source_sequence_length = max_source_sequence_length
        self.max_target_sequence_length = max_target_sequence_length

    def get_config(self):
        return {
            "vocabulary_size": self.vocabulary_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "dropout": self.dropout,
            "max_sequence_length": self.max_sequence_length,
            "name": self.name,
            "trainable": self.trainable,
        }

    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")
