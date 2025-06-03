from transformers import pipeline

sentiment_analyzer = pipeline('sentiment-analysis')

def test_text_sentiment_analysis(text):
    result = sentiment_analyzer(text)
    print(f"Sentiment analysis result: {result}")
    return result[0]

sample_text = "This too shall pass, dont be so happy."
test_text_sentiment_analysis(sample_text)
