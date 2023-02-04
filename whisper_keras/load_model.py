import os

import torch
from tensorflow import keras

from whisper_keras.whisper_backbone import WhisperBackbone

PRESETS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
}


def load_whisper_model(variant="tiny.en"):
    """
    Load a pretrained OpenAI Whisper model, convert checkpoints to Keras format,
    and return a Keras model.
    """
    if variant not in PRESETS:
        raise ValueError(
            f"Variant {variant} not found. Available variants: "
            f"{list(PRESETS.keys())}"
        )

    print("\nDownloading original OpenAI checkpoint...")
    ckpt_url = PRESETS[variant]
    checkpoint_path = keras.utils.get_file(
        fname=None,
        origin=ckpt_url,
        cache_subdir=os.path.join("checkpoint_conversion", variant),
    )

    pt_ckpt = torch.load(checkpoint_path)
    pt_cfg = pt_ckpt["dims"]
    pt_wts = pt_ckpt["model_state_dict"]

    # Form model config.
    cfg = {}

    cfg["vocabulary_size"] = pt_cfg["n_vocab"]

    assert pt_cfg["n_audio_layer"] == pt_cfg["n_text_layer"]
    cfg["num_layers"] = pt_cfg["n_audio_layer"]

    assert pt_cfg["n_audio_head"] == pt_cfg["n_text_head"]
    cfg["num_heads"] = pt_cfg["n_audio_head"]

    assert pt_cfg["n_audio_state"] == pt_cfg["n_text_state"]
    cfg["hidden_dim"] = pt_cfg["n_audio_state"]

    assert (
        pt_wts["encoder.blocks.0.mlp.0.bias"].shape[0]
        == pt_wts["decoder.blocks.0.mlp.0.bias"].shape[0]
    )
    cfg["intermediate_dim"] = pt_wts["encoder.blocks.0.mlp.0.bias"].shape[0]

    cfg["num_mels"] = pt_cfg["n_mels"]
    cfg["dropout"] = 0.0
    cfg["max_source_sequence_length"] = pt_cfg["n_audio_ctx"]
    cfg["max_target_sequence_length"] = pt_cfg["n_text_ctx"]

    print("\nModel Config:", cfg)

    print("\nInitialising Keras model...")
    model = WhisperBackbone(**cfg)
    model.summary()

    # Convert weights.
    print("\nConverting weights...")

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

    return model
