from transformers import pipeline
import speech_recognition as sr

# Use a more advanced emotion detection model (DistilRoBERTa fine-tuned for emotion detection)
emotion_analyzer = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said: ", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None

def analyze_sentiment(text):
    result = emotion_analyzer(text)
    return result[0]

if __name__ == "__main__":
    speech_text = transcribe_audio()
    if speech_text:
        emotion_result = analyze_sentiment(speech_text)
        print(f"Emotion: {emotion_result['label']}, Score: {emotion_result['score']}")
