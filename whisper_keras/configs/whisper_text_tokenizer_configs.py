"""Whisper text tokenizer configs."""

# === Multilingual configs ===

MULTILINGUAL_SPECIAL_TOKENS = {
    "bos": ("<|startoftranscript|>", 50258),
    "eos": ("<|endoftext|>", 50257),
    "pad": ("", 50256),
    "no_timestamps": ("<|notimestamps|>", 50363),
    "transcribe": ("<|transcribe|>", 50359),
    "translate": ("<|translate|>", 50358),
}

MULTILINGUAL_VOCAB_URLS = {
    "vocab_url": "https://github.com/openai/whisper/raw/main/whisper/assets/multilingual/vocab.json",
    "merges_url": "https://github.com/openai/whisper/raw/main/whisper/assets/multilingual/merges.txt",
}

MULTILINGUAL_SUPPRESSED_TOKENS = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    359,
    503,
    522,
    542,
    873,
    893,
    902,
    918,
    922,
    931,
    1350,
    1853,
    1982,
    2460,
    2627,
    3246,
    3253,
    3268,
    3536,
    3846,
    3961,
    4183,
    4667,
    6585,
    6647,
    7273,
    9061,
    9383,
    10428,
    10929,
    11938,
    12033,
    12331,
    12562,
    13793,
    14157,
    14635,
    15265,
    15618,
    16553,
    16604,
    18362,
    18956,
    20075,
    21675,
    22520,
    26130,
    26161,
    26435,
    28279,
    29464,
    31650,
    32302,
    32470,
    36865,
    42863,
    47425,
    49870,
    50254,
    50258,
    50360,
    50361,
    50362,
]

# === English configs ===

ENGLISH_SPECIAL_TOKENS = {
    "bos": ("<|startoftranscript|>", 50257),
    "eos": ("<|endoftext|>", 50256),
    "pad": ("<|endoftext|>", 50256),
    "no_timestamps": ("<|notimestamps|>", 50362),
}

ENGLISH_VOCAB_URLS = {
    "vocab_url": "https://github.com/openai/whisper/raw/main/whisper/assets/gpt2/vocab.json",
    "merges_url": "https://github.com/openai/whisper/raw/main/whisper/assets/gpt2/merges.txt",
}

ENGLISH_SUPPRESSED_TOKENS = [
    1,
    2,
    7,
    8,
    9,
    10,
    14,
    25,
    26,
    27,
    28,
    29,
    31,
    58,
    59,
    60,
    61,
    62,
    63,
    90,
    91,
    92,
    93,
    357,
    366,
    438,
    532,
    685,
    705,
    796,
    930,
    1058,
    1220,
    1267,
    1279,
    1303,
    1343,
    1377,
    1391,
    1635,
    1782,
    1875,
    2162,
    2361,
    2488,
    3467,
    4008,
    4211,
    4600,
    4808,
    5299,
    5855,
    6329,
    7203,
    9609,
    9959,
    10563,
    10786,
    11420,
    11709,
    11907,
    13163,
    13697,
    13700,
    14808,
    15306,
    16410,
    16791,
    17992,
    19203,
    19510,
    20724,
    22305,
    22935,
    27007,
    30109,
    30420,
    33409,
    34949,
    40283,
    40493,
    40549,
    47282,
    49146,
    50257,
    50359,
    50360,
    50361,
]

# === Language-related configs. ===

LANGUAGE_TO_CODE_MAPPING = {
    "english": "en",
    "chinese": "zh",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "korean": "ko",
    "french": "fr",
    "japanese": "ja",
    "portuguese": "pt",
    "turkish": "tr",
    "polish": "pl",
    "catalan": "ca",
    "dutch": "nl",
    "arabic": "ar",
    "swedish": "sv",
    "italian": "it",
    "indonesian": "id",
    "hindi": "hi",
    "finnish": "fi",
    "vietnamese": "vi",
    "hebrew": "he",
    "ukrainian": "uk",
    "greek": "el",
    "malay": "ms",
    "czech": "cs",
    "romanian": "ro",
    "danish": "da",
    "hungarian": "hu",
    "tamil": "ta",
    "norwegian": "no",
    "thai": "th",
    "urdu": "ur",
    "croatian": "hr",
    "bulgarian": "bg",
    "lithuanian": "lt",
    "latin": "la",
    "maori": "mi",
    "malayalam": "ml",
    "welsh": "cy",
    "slovak": "sk",
    "telugu": "te",
    "persian": "fa",
    "latvian": "lv",
    "bengali": "bn",
    "serbian": "sr",
    "azerbaijani": "az",
    "slovenian": "sl",
    "kannada": "kn",
    "estonian": "et",
    "macedonian": "mk",
    "breton": "br",
    "basque": "eu",
    "icelandic": "is",
    "armenian": "hy",
    "nepali": "ne",
    "mongolian": "mn",
    "bosnian": "bs",
    "kazakh": "kk",
    "albanian": "sq",
    "swahili": "sw",
    "galician": "gl",
    "marathi": "mr",
    "punjabi": "pa",
    "sinhala": "si",
    "khmer": "km",
    "shona": "sn",
    "yoruba": "yo",
    "somali": "so",
    "afrikaans": "af",
    "occitan": "oc",
    "georgian": "ka",
    "belarusian": "be",
    "tajik": "tg",
    "sindhi": "sd",
    "gujarati": "gu",
    "amharic": "am",
    "yiddish": "yi",
    "lao": "lo",
    "uzbek": "uz",
    "faroese": "fo",
    "haitian creole": "ht",
    "pashto": "ps",
    "turkmen": "tk",
    "nynorsk": "nn",
    "maltese": "mt",
    "sanskrit": "sa",
    "luxembourgish": "lb",
    "myanmar": "my",
    "tibetan": "bo",
    "tagalog": "tl",
    "malagasy": "mg",
    "assamese": "as",
    "tatar": "tt",
    "hawaiian": "haw",
    "lingala": "ln",
    "hausa": "ha",
    "bashkir": "ba",
    "javanese": "jw",
    "sundanese": "su",
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

LANGUAGE_CODE_TO_ID_MAPPING = {
    "<|af|>": 50327,
    "<|am|>": 50334,
    "<|ar|>": 50272,
    "<|as|>": 50350,
    "<|az|>": 50304,
    "<|ba|>": 50355,
    "<|be|>": 50330,
    "<|bg|>": 50292,
    "<|bn|>": 50302,
    "<|bo|>": 50347,
    "<|br|>": 50309,
    "<|bs|>": 50315,
    "<|ca|>": 50270,
    "<|cs|>": 50283,
    "<|cy|>": 50297,
    "<|da|>": 50285,
    "<|de|>": 50261,
    "<|el|>": 50281,
    "<|en|>": 50259,
    "<|es|>": 50262,
    "<|et|>": 50307,
    "<|eu|>": 50310,
    "<|fa|>": 50300,
    "<|fi|>": 50277,
    "<|fo|>": 50338,
    "<|fr|>": 50265,
    "<|gl|>": 50319,
    "<|gu|>": 50333,
    "<|haw|>": 50352,
    "<|ha|>": 50354,
    "<|he|>": 50279,
    "<|hi|>": 50276,
    "<|hr|>": 50291,
    "<|ht|>": 50339,
    "<|hu|>": 50286,
    "<|hy|>": 50312,
    "<|id|>": 50275,
    "<|is|>": 50311,
    "<|it|>": 50274,
    "<|ja|>": 50266,
    "<|jw|>": 50356,
    "<|ka|>": 50329,
    "<|kk|>": 50316,
    "<|km|>": 50323,
    "<|kn|>": 50306,
    "<|ko|>": 50264,
    "<|la|>": 50294,
    "<|lb|>": 50345,
    "<|ln|>": 50353,
    "<|lo|>": 50336,
    "<|lt|>": 50293,
    "<|lv|>": 50301,
    "<|mg|>": 50349,
    "<|mi|>": 50295,
    "<|mk|>": 50308,
    "<|ml|>": 50296,
    "<|mn|>": 50314,
    "<|mr|>": 50320,
    "<|ms|>": 50282,
    "<|mt|>": 50343,
    "<|my|>": 50346,
    "<|ne|>": 50313,
    "<|nl|>": 50271,
    "<|nn|>": 50342,
    "<|no|>": 50288,
    "<|oc|>": 50328,
    "<|pa|>": 50321,
    "<|pl|>": 50269,
    "<|ps|>": 50340,
    "<|pt|>": 50267,
    "<|ro|>": 50284,
    "<|ru|>": 50263,
    "<|sa|>": 50344,
    "<|sd|>": 50332,
    "<|si|>": 50322,
    "<|sk|>": 50298,
    "<|sl|>": 50305,
    "<|sn|>": 50324,
    "<|so|>": 50326,
    "<|sq|>": 50317,
    "<|sr|>": 50303,
    "<|su|>": 50357,
    "<|sv|>": 50273,
    "<|sw|>": 50318,
    "<|ta|>": 50287,
    "<|te|>": 50299,
    "<|tg|>": 50331,
    "<|th|>": 50289,
    "<|tk|>": 50341,
    "<|tl|>": 50348,
    "<|tr|>": 50268,
    "<|tt|>": 50351,
    "<|uk|>": 50280,
    "<|ur|>": 50290,
    "<|uz|>": 50337,
    "<|vi|>": 50278,
    "<|yi|>": 50335,
    "<|yo|>": 50325,
    "<|zh|>": 50260,
}
