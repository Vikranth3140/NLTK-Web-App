from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from collections import defaultdict

app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    input_text = request.form['inputText']
    option = request.form['option']

    # Tokenization and stopwords removal
    tokens = word_tokenize(input_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    tokenized_text = ' '.join(filtered_tokens)

    # POS Tagging
    tagged_tokens = pos_tag(filtered_tokens)
    pos_tagged_text = ' '.join([f'{token} ({tag})' for token, tag in tagged_tokens])

    # Named Entity Recognition (NER)
    ner_result = defaultdict(list)
    ne_tree = ne_chunk(pos_tagged_text.split())
    for subtree in ne_tree:
        if isinstance(subtree, nltk.tree.Tree):
            entity = ' '.join([token for token, tag in subtree.leaves()])
            label = subtree.label()
            ner_result[label].append(entity)

    if option == 'Tokenize':
        return render_template('index.html', tokenized_text=tokenized_text)
    elif option == 'PosTag':
        return render_template('index.html', pos_tagged_text=pos_tagged_text)
    elif option == 'NER':
        return render_template('index.html', ner_result=ner_result)
    elif option == 'All':
        return render_template('index.html', tokenized_text=tokenized_text, pos_tagged_text=pos_tagged_text, ner_result=ner_result)
    else:
        return render_template('index.html')  # Default render without results

if __name__ == '__main__':
    app.run(debug=True)