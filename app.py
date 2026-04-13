import os
import tempfile
import subprocess
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/corpus')
def corpus_page():
    return render_template('corpus.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    try:
        # Сохраняем файл
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Конвертируем в WAV через ffmpeg (Render имеет ffmpeg по умолчанию)
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
                raise Exception(f"Ошибка конвертации аудио: {result.stderr.decode()}")
        
        # Распознаём через Google Speech Recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        
        text = recognizer.recognize_google(audio, language='ru-RU')
        
        # Фонетическая транскрипция
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
