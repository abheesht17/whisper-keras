import tensorflow as tf

import keras_nlp
from whisper_keras.configs.whisper_text_tokenizer_configs import (
    ENGLISH_SUPPRESSED_TOKENS,
    LANGUAGE_CODE_TO_ID_MAPPING,
    LANGUAGE_TO_CODE_MAPPING,
    MULTILINGUAL_SUPPRESSED_TOKENS,
)


def generate(
    audio_features,
    model,
    tokenizer,
    language=None,
    task="transcribe",
    max_length=448,
    # no_timestamps=True,
):
    assert task in [
        "transcribe",
        "translate",
    ], f"Invalid task: {task}. Should be one of ['transcribe', 'translate']"
    assert (
        language in LANGUAGE_TO_CODE_MAPPING.keys() or language is None
    ), f"Invalid language: {language}. Should be one of {LANGUAGE_TO_CODE_MAPPING.keys()}"

    is_multilingual = tokenizer.is_multilingual
    language_ids = tf.constant(list(LANGUAGE_CODE_TO_ID_MAPPING.values()))
    suppressed_ids = tf.constant(ENGLISH_SUPPRESSED_TOKENS)

    decoder_token_ids = tf.constant([[tokenizer.bos_id]])
    if is_multilingual:
        suppressed_ids = tf.constant(MULTILINGUAL_SUPPRESSED_TOKENS)

        if language is None:
            # Detect the language.

            def token_probability_fn(prompt):
                inputs = {
                    "encoder_features": audio_features,
                    "encoder_padding_mask": tf.ones_like(
                        audio_features[:, :1500, 0]
                    ),
                    "decoder_token_ids": prompt,
                    "decoder_padding_mask": tf.zeros_like(prompt),
                }
                logits = model(inputs)[:, -1, :]
                vocabulary_size = tf.shape(logits)[-1]

                # Mask out all tokens except language tokens.
                mask = tf.reduce_any(
                    tf.equal(
                        tf.expand_dims(tf.range(vocabulary_size), axis=0),
                        tf.expand_dims(language_ids, axis=1),
                    ),
                    axis=0,
                )
                logits = tf.where(mask, logits, tf.fill(tf.shape(logits), -1e9))

                return logits

            decoder_token_ids = keras_nlp.utils.greedy_search(
                token_probability_fn=token_probability_fn,
                prompt=decoder_token_ids,
                max_length=2,
            )
        else:
            language_code = LANGUAGE_TO_CODE_MAPPING[language]
            language_id = LANGUAGE_CODE_TO_ID_MAPPING[f"<{language_code}>"]
            decoder_token_ids = tf.concat(
                (decoder_token_ids, [[language_id]]), axis=1
            )

        if task == "transcribe":
            decoder_token_ids = tf.concat(
                (decoder_token_ids, [[tokenizer.transcribe_id]]), axis=1
            )
        else:
            decoder_token_ids = tf.concat(
                (decoder_token_ids, [[tokenizer.translate_id]]), axis=1
            )

    decoder_token_ids = tf.concat(
        (decoder_token_ids, [[tokenizer.no_timestamps_id]]), axis=1
    )

    def token_probability_fn(prompt):
        inputs = {
            "encoder_features": audio_features,
            "encoder_padding_mask": tf.ones_like(audio_features[:, :1500, 0]),
            "decoder_token_ids": prompt,
            "decoder_padding_mask": tf.ones_like(prompt),
        }
        logits = model(inputs)[:, -1, :]
        vocabulary_size = tf.shape(logits)[-1]

        # Mask out all tokens except language tokens.
        mask = tf.reduce_any(
            tf.not_equal(
                tf.expand_dims(tf.range(vocabulary_size), axis=0),
                tf.expand_dims(suppressed_ids, axis=1),
            ),
            axis=0,
        )
        logits = tf.where(mask, logits, tf.fill(tf.shape(logits), -1e9))
        return logits

    decoder_token_ids = keras_nlp.utils.greedy_search(
        token_probability_fn=token_probability_fn,
        prompt=decoder_token_ids,
        max_length=max_length,
        end_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
    )
    tf.print(decoder_token_ids)

    decoder_sentences = tokenizer.detokenize(decoder_token_ids)

    return decoder_sentences
