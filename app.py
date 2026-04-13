import os
import json
import tempfile
import subprocess
import wave
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import vosk

app = Flask(__name__)
CORS(app)

# Путь к модели Vosk
MODEL_PATH = "vosk-model-small-ru-0.22"

# Загружаем модель при старте
if os.path.exists(MODEL_PATH):
    model = vosk.Model(MODEL_PATH)
    print(f"✅ Модель Vosk загружена из {MODEL_PATH}")
else:
    model = None
    print(f"⚠️ Модель не найдена: {MODEL_PATH}")

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

def convert_to_wav(input_path):
    """Конвертация аудио в WAV 16kHz mono"""
    output_path = tempfile.mktemp(suffix='.wav')
    cmd = [
        'ffmpeg', '-i', input_path,
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y', output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise Exception(f"Ошибка конвертации: {result.stderr.decode()}")
    return output_path

def transcribe_audio(audio_path):
    """Распознавание речи через Vosk"""
    if model is None:
        return "Модель не загружена"
    
    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 48000]:
        raise Exception("Аудио должно быть mono и с поддерживаемой частотой дискретизации")
    
    rec = vosk.KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(False)
    
    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text += " " + res.get("text", "")
    
    final_res = json.loads(rec.FinalResult())
    text += " " + final_res.get("text", "")
    
    return text.strip()

def text_to_phonetic(text):
    """Преобразование в фонетическую транскрипцию"""
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

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    try:
        # Сохраняем загруженный файл
        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else 'wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Конвертируем в WAV (если не WAV)
        if not ext == 'wav':
            wav_path = convert_to_wav(tmp_path)
            os.unlink(tmp_path)
        else:
            wav_path = tmp_path
        
        # Распознаём
        text = transcribe_audio(wav_path)
        
        # Если распознавание не дало результата
        if not text:
            text = "[речь не распознана]"
        
        # Фонетическая транскрипция
        phonetic = text_to_phonetic(text)
        
        # Очистка
        os.unlink(wav_path)
        
        return jsonify({
            'orthographic': text,
            'phonetic': phonetic
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
