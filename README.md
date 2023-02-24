# WhisperKeras

WhisperKeras is a project which implements [OpenAI's Whisper](https://openai.com/blog/whisper/)
model in Keras. The Whisper model is a speech-to-text transformer-based
encoder-decoder model.

## Quick Start

For a quick start, you can try transcribing audio using [this Colab notebook](https://colab.research.google.com/drive/1SjKogodXfQwZvNnfBU6Mlkn9NN1w7WFQ?usp=sharing).

## Setup

Clone the repository:

```sh
git clone https://github.com/abheesht17/whisper-keras.git
cd whisper-keras
```

After cloning the repo, install all the necessary dependencies:

```sh
pip install -r requirements.txt --upgrade
```

Add the repository to `PYTHONPATH` so that we can easily import and use the
library. You can either do this using a Linux shell command, or you can do it
easily inside your Python script like this:

```python
import sys
sys.path.append("/path/to/the/repo")
```

Import the library and you are good to go:

```python
import whisper_keras
```

## Transcribe your own audio!

In this section, we will demonstrate the various settings we can use to
generate audio. Some sample audio files are present [here](https://github.com/abheesht17/whisper-keras).

### English-only Preset

There are multiple Whisper presets available. We will go with `"tiny.en"`, an
English-only model.

```
mp3_path = "./audio_samples/english.mp3"
preset = "tiny.en"

# Load the audio file and compute the log-mel spectrogram.
audio = whisper_keras.load_audio(mp3_path)
whisper_audio_feature_extractor = whisper_keras.WhisperAudioFeatureExtractor()
audio_features = whisper_audio_feature_extractor(audio)

# Load the preset tokenizer and model.
tokenizer = whisper_keras.load_tokenizer(preset)
model = whisper_keras.load_model(preset)

# Transcribe the audio!
whisper_keras.generate(
    audio_features=audio_features,
    model=model,
    tokenizer=tokenizer,
    remove_pad_tokens=True,
)
```

Output:
```
array([b'<|startoftranscript|><|notimestamps|> My dear Fanny, you feel these things a great deal too much. I am most happy that you like the chain.'],
      dtype=object)>
```

### Multilingual Models

#### Let the model detect the language!

We will use the `"tiny"` preset.

```
mp3_path = "./audio_samples/french.mp3"
preset = "tiny"

# Load the audio file and compute the log-mel spectrogram.
audio = whisper_keras.load_audio(mp3_path)
whisper_audio_feature_extractor = whisper_keras.WhisperAudioFeatureExtractor()
audio_features = whisper_audio_feature_extractor(audio)

# Load the preset tokenizer and model.
tokenizer = whisper_keras.load_tokenizer(preset)
model = whisper_keras.load_model(preset)

# Transcribe the audio!
whisper_keras.generate(
    audio_features=audio_features,
    model=model,
    tokenizer=tokenizer,
    remove_pad_tokens=True,
)
```

Output:
```
tf.Tensor([b"<|startoftranscript|><|fr|><|transcribe|><|notimestamps|> Aujourd'hui, je vais vous parler du Pire cadeau d'anniversaire que j'ai re\xc3\xa7u. Quand j'\xc3\xa9tais \xc3\xa0 D\xc3\xb4me, on perd de m'en demande des toujours de faire une liste pour mes cadeaux et la plupart du temps quand je les re\xc3\xa7ovi\xc3\xa9, il y avait \xc3\xa9norm\xc3\xa9ment de choses qui ne faisaient pas partie de cette fameuse liste et qui ne me correspondaient pas forc\xc3\xa9ment et c'\xc3\xa9tait le cas lors de cette anniversaire. J'ai re\xc3\xa7u un petit sac<|endoftext|>"], shape=(1,), dtype=string)
```


If you want the target language to be English, you can pass `translate = True` to
the `generate()` function.

```
generated_text = whisper_keras.generate(
    audio_features=audio_features,
    model=model,
    tokenizer=tokenizer,
    task="translate",
    remove_pad_tokens=True,
)
```

Output:
```
tf.Tensor([b"<|startoftranscript|><|fr|><|translate|><|notimestamps|> Today I will talk about Pire Cado de Niversaire when I was at home, my father asked me to do a list for my Cado and the most part of the time when I was going to be able to receive him. He had a lot of things that he didn't do with this famous list and he didn't answer his questions and it was the case. Then this year, I received a little bit of that.<|endoftext|>"], shape=(1,), dtype=string)
```

#### If you are sure about the source language, hardcode it!

```
generated_text = whisper_keras.generate(
    audio_features=audio_features,
    model=model,
    tokenizer=tokenizer,
    language="french",
    task="translate",
    remove_pad_tokens=True,
)
```

### To-dos [24/02/2023]

- [ ] Support decoding methods other than greedy search.
- [ ] Enable batch decoding.
- [ ] Add sliding window technique so that audio longer than 30 seconds can be transcribed.
- [ ] Make text generation faster and more efficient.
