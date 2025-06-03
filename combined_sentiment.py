from voice_analysis.speech_to_text import transcribe_audio, analyze_sentiment as analyze_audio_sentiment
from face_analysis.face_detection import detect_face_emotion
from text_analysis.sentiment_analysis import analyze_sentiment as analyze_text_sentiment

def get_combined_sentiment():
    # Get sentiment from speech (audio-based)
    speech_text = transcribe_audio()
    audio_sentiment = None
    if speech_text:
        audio_sentiment = analyze_audio_sentiment(speech_text)
        print(f"Audio Sentiment: {audio_sentiment['label']}, Score: {audio_sentiment['score']}")

    # Get sentiment from face emotion detection (face-based)
    face_emotion = detect_face_emotion()
    face_sentiment_score = {
        'happy': 1,
        'surprised': 0.8,
        'neutral': 0,  # Neutral emotion gets 0
        'angry': -1,
        'sad': -0.8,
        'fear': -0.5,
        'disgust': -0.6,
        'surprise': 0.7
    }.get(face_emotion, 0)  # Default to 0 if no recognized emotion
    print(f"Face Sentiment: {face_emotion}, Score: {face_sentiment_score}")

    # Get sentiment from text (text-based)
    text_sentiment = None
    if speech_text:
        text_sentiment = analyze_text_sentiment(speech_text)
        print(f"Text Sentiment: {text_sentiment['label']}, Score: {text_sentiment['score']}")

    # Combine the results
    # Normalize the scores to be between -1 and 1
    audio_score = 1 if audio_sentiment and audio_sentiment['label'] == "POSITIVE" else -1
    face_score = face_sentiment_score
    text_score = text_sentiment['score'] if text_sentiment else 0  # Use 0 if no text sentiment is available

    # Calculate combined score (weighted average)
    combined_score = (audio_score + face_score + text_score) / 3
    combined_label = "POSITIVE" if combined_score > 0 else "NEGATIVE"

    print(f"Combined Sentiment: {combined_label}, Combined Score: {combined_score}")

if __name__ == "__main__":
    get_combined_sentiment()
