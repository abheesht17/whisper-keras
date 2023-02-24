import os

import torch
from tensorflow import keras

from whisper_keras.configs.whisper_presets import PRESETS
from whisper_keras.configs.whisper_text_tokenizer_configs import (
    ENGLISH_SPECIAL_TOKENS,
    ENGLISH_VOCAB_URLS,
    LANGUAGE_CODE_TO_ID_MAPPING,
    MULTILINGUAL_SPECIAL_TOKENS,
    MULTILINGUAL_VOCAB_URLS,
)
from whisper_keras.models.whisper_backbone import WhisperBackbone
from whisper_keras.models.whisper_speech_to_text import WhisperSpeechToText
from whisper_keras.preprocessors.whisper_text_tokenizer import (
    WhisperTextTokenizer,
)


def _raise_preset_not_found_error(preset):
    if preset not in PRESETS:
        raise ValueError(
            f"Variant {preset} not found. Available presets: "
            f"{list(PRESETS.keys())}"
        )


def load_model(preset="tiny.en"):
    """
    Load a pretrained OpenAI Whisper model, convert checkpoints to Keras format,
    and return a `whisper_keras.models.WhisperSpeechToText` instance.
    """
    _raise_preset_not_found_error(preset)

    # Download original OpenAI checkpoint.
    ckpt_url = PRESETS[preset]["weights_url"]
    checkpoint_path = keras.utils.get_file(
        fname=None,
        origin=ckpt_url,
        cache_subdir=os.path.join("whisper_keras_presets", preset, "model.pt"),
    )

    pt_ckpt = torch.load(checkpoint_path)
    pt_wts = pt_ckpt["model_state_dict"]

    # Fetch model config.
    cfg = PRESETS[preset]["model_config"]

    # Initialize Keras model.
    model = WhisperBackbone(**cfg)

    # Convert weights.

    # ===== Encoder weights =====

    # === Encoder embedding layer ===
    model.get_layer("encoder_token_embedding_conv_layer_1").kernel.assign(
        pt_wts["encoder.conv1.weight"].permute(2, 1, 0).numpy()
    )
    model.get_layer("encoder_token_embedding_conv_layer_1").bias.assign(
        pt_wts["encoder.conv1.bias"].numpy()
    )
    model.get_layer("encoder_token_embedding_conv_layer_2").kernel.assign(
        pt_wts["encoder.conv2.weight"].permute(2, 1, 0).numpy()
    )
    model.get_layer("encoder_token_embedding_conv_layer_2").bias.assign(
        pt_wts["encoder.conv2.bias"].numpy()
    )

    model.get_layer("encoder_position_embedding").position_embeddings.assign(
        pt_wts["encoder.positional_embedding"]
    )

    # === Encoder Transformer Layers ===
    for i in range(model.num_layers):
        # === Self-attention ===
        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            pt_wts[f"encoder.blocks.{i}.attn.query.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            pt_wts[f"encoder.blocks.{i}.attn.query.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            pt_wts[f"encoder.blocks.{i}.attn.key.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )

        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            pt_wts[f"encoder.blocks.{i}.attn.value.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            pt_wts[f"encoder.blocks.{i}.attn.value.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            pt_wts[f"encoder.blocks.{i}.attn.out.weight"]
            .transpose(1, 0)
            .reshape((cfg["num_heads"], -1, cfg["hidden_dim"]))
            .numpy()
        )
        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            pt_wts[f"encoder.blocks.{i}.attn.out.bias"].numpy()
        )

        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layernorm.gamma.assign(
            pt_wts[f"encoder.blocks.{i}.attn_ln.weight"].numpy()
        )
        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layernorm.beta.assign(
            pt_wts[f"encoder.blocks.{i}.attn_ln.bias"].numpy()
        )

        # === Intermediate and output MLP layers ===
        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            pt_wts[f"encoder.blocks.{i}.mlp.0.weight"].transpose(1, 0).numpy()
        )
        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            pt_wts[f"encoder.blocks.{i}.mlp.0.bias"].numpy()
        )

        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            pt_wts[f"encoder.blocks.{i}.mlp.2.weight"].transpose(1, 0).numpy()
        )
        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            pt_wts[f"encoder.blocks.{i}.mlp.2.bias"].numpy()
        )

        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_layernorm.gamma.assign(
            pt_wts[f"encoder.blocks.{i}.mlp_ln.weight"].numpy()
        )
        model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_layernorm.beta.assign(
            pt_wts[f"encoder.blocks.{i}.mlp_ln.bias"].numpy()
        )

    # === Final LayerNorm ===
    model.get_layer("encoder_layer_norm").gamma.assign(
        pt_wts["encoder.ln_post.weight"]
    )
    model.get_layer("encoder_layer_norm").beta.assign(
        pt_wts["encoder.ln_post.bias"]
    )

    # ===== Decoder weights =====

    # === Decoder embedding layer ===
    model.get_layer(
        "decoder_token_and_position_embedding"
    ).token_embedding.embeddings.assign(
        pt_wts["decoder.token_embedding.weight"]
    )
    model.get_layer(
        "decoder_token_and_position_embedding"
    ).position_embedding.position_embeddings.assign(
        pt_wts["decoder.positional_embedding"]
    )

    # === Decoder Transformer Layers ===
    for i in range(model.num_layers):
        # === Self-attention ===
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.attn.query.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            pt_wts[f"decoder.blocks.{i}.attn.query.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.attn.key.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.attn.value.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            pt_wts[f"decoder.blocks.{i}.attn.value.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.attn.out.weight"]
            .transpose(1, 0)
            .reshape((cfg["num_heads"], -1, cfg["hidden_dim"]))
            .numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            pt_wts[f"decoder.blocks.{i}.attn.out.bias"].numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layernorm.gamma.assign(
            pt_wts[f"decoder.blocks.{i}.attn_ln.weight"].numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layernorm.beta.assign(
            pt_wts[f"decoder.blocks.{i}.attn_ln.bias"].numpy()
        )

        # === Cross-attention ===
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._query_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.cross_attn.query.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._query_dense.bias.assign(
            pt_wts[f"decoder.blocks.{i}.cross_attn.query.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._key_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.cross_attn.key.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._value_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.cross_attn.value.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._value_dense.bias.assign(
            pt_wts[f"decoder.blocks.{i}.cross_attn.value.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._output_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.cross_attn.out.weight"]
            .transpose(1, 0)
            .reshape((cfg["num_heads"], -1, cfg["hidden_dim"]))
            .numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._output_dense.bias.assign(
            pt_wts[f"decoder.blocks.{i}.cross_attn.out.bias"].numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layernorm.gamma.assign(
            pt_wts[f"decoder.blocks.{i}.cross_attn_ln.weight"].numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layernorm.beta.assign(
            pt_wts[f"decoder.blocks.{i}.cross_attn_ln.bias"].numpy()
        )

        # === Intermediate and output MLP layers ===
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.mlp.0.weight"].transpose(1, 0).numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            pt_wts[f"decoder.blocks.{i}.mlp.0.bias"].numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            pt_wts[f"decoder.blocks.{i}.mlp.2.weight"].transpose(1, 0).numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            pt_wts[f"decoder.blocks.{i}.mlp.2.bias"].numpy()
        )

        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_layernorm.gamma.assign(
            pt_wts[f"decoder.blocks.{i}.mlp_ln.weight"].numpy()
        )
        model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_layernorm.beta.assign(
            pt_wts[f"decoder.blocks.{i}.mlp_ln.bias"].numpy()
        )

    # === Final LayerNorm ===
    model.get_layer("decoder_layer_norm").gamma.assign(
        pt_wts["decoder.ln.weight"]
    )
    model.get_layer("decoder_layer_norm").beta.assign(pt_wts["decoder.ln.bias"])

    # WhisperSpeechToText
    model = WhisperSpeechToText(backbone=model)

    return model


def load_tokenizer(preset="tiny.en"):
    """
    Loads the appropriate vocabulary, merges, special tokens, etc. and returns
    a `whisper_keras.preprocessors.WhisperTextTokenizer` instance.
    """
    _raise_preset_not_found_error(preset)

    if PRESETS[preset]["is_multilingual"]:
        vocab_url = MULTILINGUAL_VOCAB_URLS["vocab_url"]
        merges_url = MULTILINGUAL_VOCAB_URLS["merges_url"]
        special_tokens_dict = MULTILINGUAL_SPECIAL_TOKENS
        language_tokens_dict = LANGUAGE_CODE_TO_ID_MAPPING
    else:
        vocab_url = ENGLISH_VOCAB_URLS["vocab_url"]
        merges_url = ENGLISH_VOCAB_URLS["merges_url"]
        special_tokens_dict = ENGLISH_SPECIAL_TOKENS
        language_tokens_dict = None

    vocab_path = keras.utils.get_file(
        fname=None,
        origin=vocab_url,
        cache_subdir=os.path.join(
            "whisper_keras_presets", preset, "vocab.json"
        ),
    )
    merges_path = keras.utils.get_file(
        fname=None,
        origin=merges_url,
        cache_subdir=os.path.join(
            "whisper_keras_presets", preset, "merges.txt"
        ),
    )

    # Define the tokenizer.
    tokenizer = WhisperTextTokenizer(
        vocabulary=vocab_path,
        merges=merges_path,
        special_tokens_dict=special_tokens_dict,
        language_tokens_dict=language_tokens_dict,
        is_multilingual=PRESETS[preset]["is_multilingual"],
    )
    return tokenizer
