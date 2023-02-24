"""Whisper tokenizer."""

import json

from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer


class WhisperTextTokenizer(BytePairTokenizer):
    """Whisper text tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.BytePairTokenizer`.
    This tokenizer does not provide truncation or padding of inputs.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.
        special_tokens_dict: dict, containing special tokens such as BOS, EOS,
            etc. For example, check
            `whisper_keras.configs.whisper_text_tokenizer_configs.MULTILINGUAL_SPECIAL_TOKENS`.
        language_tokens_dict: dict, containing language tokens, with token as key
            and token ID as value. For example, check
            `whisper_keras.configs.whisper_text_tokenizer_configs.LANGUAGE_CODE_TO_ID_MAPPING`.
        is_multilingual: bool, whether the tokenizer is multilingual or not.
    """

    def __init__(
        self,
        vocabulary,
        merges,
        special_tokens_dict,
        language_tokens_dict=None,
        is_multilingual=False,
        **kwargs,
    ):
        # The vocabulary files do not have special tokens. We will insert it in the
        # vocabulary. Note that HuggingFace and the original OpenAI implementation
        # do not add them to the vocabulary; they treat them separately as
        # "added_tokens". But adding them to the vocabulary is more convenient
        # during detokenization (without affecting tokenization). At the same time,
        # we add special tokens as member variables.
        if isinstance(vocabulary, str):
            with open(vocabulary, "r") as f:
                vocabulary = json.load(f)

        for token_type in special_tokens_dict:
            token = special_tokens_dict[token_type][0]
            token_idx = special_tokens_dict[token_type][1]
            vocabulary[token] = token_idx

            # Add special tokens as member variables.
            setattr(self, f"{token_type}_token", token)
            setattr(self, f"{token_type}_id", token_idx)

        # Add language tokens to the vocabulary.
        if language_tokens_dict is not None:
            for language_token in language_tokens_dict:
                language_token_id = language_tokens_dict[language_token]
                vocabulary[language_token] = language_token_id

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

        self.is_multilingual = is_multilingual
