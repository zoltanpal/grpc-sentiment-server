
# Sentiment Analyzer Library

This Python library provides sentiment analysis for multiple languages using different models. It currently supports:

- Hungarian (hun): Uses a fine-tuned RoBERTa model.
- Danish (dan): Uses a fine-tuned transformer model.
- English (eng): Uses the VADER sentiment analysis model.

## Features

✅ Supports multiple languages with different models  
✅ Uses singleton pattern for efficient model loading  
✅ Thread-safe and optimized for concurrent API calls  
✅ Automatically downloads missing dependencies (e.g., VADER lexicon)

## Installation
Ensure you have the necessary dependencies installed:
```commandline
pip install transformers torch nltk
```
For the first-time setup, ensure NLTK has the required lexicon for English:
```python
import nltk
nltk.download('vader_lexicon')
```

## Folder Structure
```easycode
sentiment_analyzers/
│── analyzers/
│   ├── base_analyzer.py
│   ├── hun/
│   │   ├── sentiment_analyzer.py
│   ├── dan/
│   │   ├── sentiment_analyzer.py
│   ├── eng/
│   │   ├── sentiment_analyzer.py
│── factory/
│   ├── sentiment_factory.py
│── models/
│   ├── sentiments.py
```

## Usage
You can use the **SentimentAnalyzerFactory** to get the appropriate sentiment analyzer instance based on the language.

Example:
```python
from sentiment_analyzers.factory.sentiment_factory import SentimentAnalyzerFactory

# Get Hungarian sentiment analyzer
hun_analyzer = SentimentAnalyzerFactory.get_analyzer("hun")
result_hu = hun_analyzer.analyze_text("Ez egy fantasztikus film volt!")
print(result_hu)

# Get Danish sentiment analyzer
dan_analyzer = SentimentAnalyzerFactory.get_analyzer("dan")
result_da = dan_analyzer.analyze_text("Det var en rigtig god film!")
print(result_da)

# Get English sentiment analyzer
en_analyzer = SentimentAnalyzerFactory.get_analyzer("eng")
result_en = en_analyzer.analyze_text("This was an amazing movie!")
print(result_en)
```

## Adding More Languages

To add a new language:
- Create a new folder under ```analyzers/``` (e.g., analyzers/fr/ for French).
- Implement a ```sentiment_analyzer.py``` similar to Hungarian or Danish.
- Register the new analyzer in ```sentiment_factory.py```.


## Contributions

Feel free to fork, submit issues, or contact me to improve the library!