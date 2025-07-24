from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Download and cache locally
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
processor = Wav2Vec2FeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
model.save_pretrained("./local-wav2vec-model")

processor.save_pretrained("./local-wav2vec-model")

print("complete")
