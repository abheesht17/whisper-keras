import tensorflow as tf

from keras_nlp.models.task import Task


class WhisperSpeechToText(Task):
    def __init__(
        self,
        backbone,
        **kwargs,
    ):
        inputs = backbone.input
        x = backbone(inputs)
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
