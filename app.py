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

# Хранилище сессий
sessions = {}

# Правила для фонетической транскрипции
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
    """Преобразование в фонетическую транскрипцию"""
    if not text:
        return ""
    
    text = text.lower().strip()
    phonetic = []
    
    i = 0
    while i < len(text):
        char = text[i]
        if i < len(text) - 1:
            two = text[i:i+2]
            if two == 'тс':
                phonetic.append('ts')
                i += 2
                continue
            elif two == 'дж':
                phonetic.append('dʐ')
                i += 2
                continue
        
        if char in LETTER_TO_PHONEME:
            phoneme = LETTER_TO_PHONEME[char]
            if phoneme:
                phonetic.append(phoneme)
        elif char == ' ':
            phonetic.append(' ')
        else:
            phonetic.append(char)
        i += 1
    
    result = ''.join(phonetic)
    words = result.split()
    voiced = {'b': 'p', 'v': 'f', 'g': 'k', 'd': 't', 'ʐ': 'ʂ', 'z': 's'}
    
    processed = []
    for word in words:
        if word and word[-1] in voiced:
            word = word[:-1] + voiced[word[-1]]
        processed.append(word)
    
    return ' '.join(processed)


# ==================== МАРШРУТЫ ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/corpus')
def corpus_page():
    return render_template('corpus.html')


# ==================== КОРПУС: ЗАГРУЗКА И ПАРСИНГ ====================

@app.route('/upload_corpus', methods=['POST'])
def upload_corpus():
    if 'files' not in request.files:
        return jsonify({'error': 'Нет файлов'}), 400
    
    files = request.files.getlist('files')
    parsed_manuscripts = []
    
    for file in files:
        if file.filename.endswith('.xml'):
            try:
                tree = ET.parse(file)
                root = tree.getroot()
                
                verses = {}
                
                # Пробуем с namespace и без
                for ab in root.findall('.//ab[@n]'):
                    verse_num = ab.get('n')
                    verse_text = ' '.join(ab.itertext()).strip()
                    if verse_num and verse_text:
                        verses[verse_num] = verse_text
                
                if not verses:
                    for ab in root.findall('.//{http://www.tei-c.org/ns/1.0}ab[@n]'):
                        verse_num = ab.get('n')
                        verse_text = ' '.join(ab.itertext()).strip()
                        if verse_num and verse_text:
                            verses[verse_num] = verse_text
                
                parsed_manuscripts.append({
                    'name': file.filename,
                    'verses': verses,
                    'verse_count': len(verses)
                })
            except Exception as e:
                return jsonify({'error': f'Ошибка парсинга {file.filename}: {str(e)}'}), 400
    
    return jsonify({'manuscripts': parsed_manuscripts})


# ==================== КОРПУС: СОХРАНЕНИЕ И ЗАГРУЗКА СЕССИЙ ====================

@app.route('/save_session', methods=['POST'])
def save_session():
    data = request.json
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = data
    return jsonify({'session_id': session_id})

@app.route('/load_session/<session_id>', methods=['GET'])
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
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    try:
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        wav_path = tmp_path
        if ext != 'wav':
            wav_path = tempfile.mktemp(suffix='.wav')
            result = subprocess.run([
                'ffmpeg', '-i', tmp_path,
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', wav_path
            ], capture_output=True, timeout=30)
            os.unlink(tmp_path)
            
            if result.returncode != 0:
                raise Exception(f"Ошибка конвертации аудио")
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        
        text = recognizer.recognize_google(audio, language='ru-RU')
        phonetic = text_to_phonetic(text)
        
        os.unlink(wav_path)
        
        return jsonify({
            'orthographic': text,
            'phonetic': phonetic
        })
    
    except sr.UnknownValueError:
        return jsonify({'error': 'Речь не распознана. Попробуйте другой файл.'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Ошибка сервиса распознавания: {e}'}), 500
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Превышено время обработки аудио'}), 500
    except Exception as e:
        return jsonify({'error': f'Внутренняя ошибка: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
