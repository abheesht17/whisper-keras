"""Whisper tokenizer."""

import json

from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer


class WhisperTextTokenizer(BytePairTokenizer):
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
            setattr(self, token_type, token)
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
