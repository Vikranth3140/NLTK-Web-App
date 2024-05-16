from flask import Flask, render_template, request

app = Flask(__name__)

# Import NLTK and download necessary resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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