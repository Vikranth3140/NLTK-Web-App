from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    input_text = request.form['inputText']
    tokens = word_tokenize(input_text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    tokenized_text = ' '.join(filtered_tokens)
    return render_template('index.html', tokenized_text=tokenized_text)

if __name__ == '__main__':
    app.run(debug=True)