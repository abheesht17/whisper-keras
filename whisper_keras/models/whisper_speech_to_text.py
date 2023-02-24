import tensorflow as tf

from keras_nlp.models.task import Task


class WhisperSpeechToText(Task):
    """Whisper Speech-to-Text encoder-decoder model.

    Adds the output logits layer on top of `WhisperBackbone`. This output logits
    layer is the transposed decoder embedding layer, that is, the shape is
    is `(hidden_size, vocabulary_size)`.

    Args:
        backbone: A `WhisperBackbone` instance.
    """

    def __init__(
        self,
        backbone,
        **kwargs,
    ):
        inputs = backbone.input
        x = backbone(inputs)["decoder_sequence_output"]
        # Use token embedding weights to project from the token representation
        # to vocabulary logits.
        outputs = tf.matmul(
            x,
            backbone.decoder_token_embedding.embeddings,
            transpose_b=True,
        )

        # Instantiate using Functional API Model constructor.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=False,
            **kwargs,
        )

        self.backbone = backbone
