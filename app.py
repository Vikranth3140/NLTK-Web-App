from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['inputText']
    option = request.form['option']

    if not input_text.strip():  # Check if input text is empty or contains only whitespace
        return render_template('index.html', error_message='Input text cannot be empty.')

    # Tokenization and stopwords removal
    tokens = word_tokenize(input_text)
    if not tokens:  # Check if tokenization produced any tokens
        return render_template('index.html', error_message='Error in tokenization.')

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    tokenized_text = ' '.join(filtered_tokens)

    # POS Tagging
    tagged_tokens = pos_tag(filtered_tokens)
    pos_tagged_text = ' '.join([f'{token} ({tag})' for token, tag in tagged_tokens])

    # Named Entity Recognition (NER)
    ner_result = defaultdict(list)
    ne_tree = ne_chunk(tagged_tokens)
    for subtree in ne_tree:
        if isinstance(subtree, nltk.tree.Tree):
            entity = ' '.join([token for token, tag in subtree.leaves()])
            label = subtree.label()
            ner_result[label].append(entity)

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(input_text)

    if option == 'Tokenize':
        return render_template('index.html', tokenized_text=tokenized_text)
    elif option == 'PosTag':
        return render_template('index.html', pos_tagged_text=pos_tagged_text)
    elif option == 'NER':
        return render_template('index.html', ner_result=ner_result)
    elif option == 'Sentiment':
        return render_template('index.html', sentiment_scores=sentiment_scores)
    elif option == 'All':
        return render_template('index.html', tokenized_text=tokenized_text, pos_tagged_text=pos_tagged_text, ner_result=ner_result, sentiment_scores=sentiment_scores)
    else:
        return render_template('index.html')  # Default render without results

if __name__ == '__main__':
    app.run(debug=True)