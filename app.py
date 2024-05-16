from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['inputText']
    
    # Tokenization and stopwords removal
    tokens = word_tokenize(input_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    tokenized_text = ' '.join(filtered_tokens)
    
    # POS Tagging
    tagged_tokens = pos_tag(tokens)
    pos_tagged_text = ' '.join([f'{token} ({tag})' for token, tag in tagged_tokens])
    
    return render_template('index.html', tokenized_text=tokenized_text, pos_tagged_text=pos_tagged_text)

if __name__ == '__main__':
    app.run(debug=True)