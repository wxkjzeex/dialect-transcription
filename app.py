import os
import tempfile
import uuid
import subprocess
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app)

# ==================== СЕССИИ ====================
sessions = {}

# ==================== ФОНЕТИКА ====================

LETTER_TO_PHONEME = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
    'е': 'je', 'ё': 'jo', 'ж': 'ʐ', 'з': 'z', 'и': 'i',
    'й': 'j', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
    'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
    'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'tɕ',
    'ш': 'ʂ', 'щ': 'ɕː', 'ъ': '', 'ы': 'ɨ', 'ь': 'ʲ',
    'э': 'e', 'ю': 'ju', 'я': 'ja'
}

def text_to_phonetic(text):
    if not text:
        return ""

    text = text.lower().strip()
    result = []

    for word in text.split():
        phon = []
        i = 0
        while i < len(word):
            if i < len(word) - 1:
                two = word[i:i+2]
                if two == 'тс':
                    phon.append('ts')
                    i += 2
                    continue
                if two == 'дж':
                    phon.append('dʐ')
                    i += 2
                    continue

            char = word[i]
            phoneme = LETTER_TO_PHONEME.get(char, char)
            if phoneme:
                phon.append(phoneme)
            i += 1

        # оглушение
        voiced = {'b': 'p', 'v': 'f', 'g': 'k', 'd': 't', 'ʐ': 'ʂ', 'z': 's'}
        if phon and phon[-1] in voiced:
            phon[-1] = voiced[phon[-1]]

        result.append(''.join(phon))

    return ' '.join(result)


# ==================== TEI ПАРСИНГ ====================

def parse_tei(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    words = []

    # ✅ ВАЖНО: идём по документу в правильном порядке
    for el in root.iter():
        tag = el.tag.split('}')[-1]

        if tag in ['w', 'pc']:
            text = ''.join(el.itertext()).strip()
            if text:
                words.append(text)

    full_text = ' '.join(words)

    return {
        'words': words,
        'fullText': full_text
    }


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/corpus')
def corpus_page():
    return render_template('corpus.html')


# ==================== UPLOAD ====================

@app.route('/upload_corpus', methods=['POST'])
def upload_corpus():
    if 'files' not in request.files:
        return jsonify({'error': 'Нет файлов'}), 400

    files = request.files.getlist('files')
    result = []

    for file in files:
        if not file.filename.endswith('.xml'):
            continue

        try:
            data = parse_tei(file)

            result.append({
                'name': file.filename,
                'words': data['words'],
                'fullText': data['fullText'],
                'word_count': len(data['words'])
            })

        except Exception as e:
            return jsonify({'error': f'Ошибка в {file.filename}: {str(e)}'}), 400

    return jsonify({'manuscripts': result})


# ==================== СЕССИИ ====================

@app.route('/save_session', methods=['POST'])
def save_session():
    data = request.json
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = data
    return jsonify({'session_id': session_id})


@app.route('/load_session/<session_id>')
def load_session(session_id):
    if session_id in sessions:
        return jsonify(sessions[session_id])
    return jsonify({'error': 'Сессия не найдена'}), 404


# ==================== ТРАНСКРИПЦИЯ ====================

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400

    file = request.files['audio']

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            wav_path = tmp.name

        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language='ru-RU')
        phonetic = text_to_phonetic(text)

        os.unlink(wav_path)

        return jsonify({
            'text': text,
            'phonetic': phonetic
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== RUN ====================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
