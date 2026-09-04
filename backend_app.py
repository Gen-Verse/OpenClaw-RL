from flask import Flask, request, jsonify
from style_analyzer import analyze_directory, generate_sample

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    directory = data.get('dir')
    if not directory:
        return jsonify({'error': 'missing dir parameter'}), 400
    analysis = analyze_directory(directory)
    sample = generate_sample(analysis)
    return jsonify({'analysis': analysis, 'sample': sample})

if __name__ == '__main__':
    # Flask default server; for production, run with a WSGI server
    app.run(host='127.0.0.1', port=5000)
