import os
import json
import wave
import subprocess
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)


# Проверяем наличие ffmpeg
def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except:
        return False


# Правила преобразования текста в фонетическую транскрипцию
LETTER_TO_PHONEME = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
    'е': 'je', 'ё': 'jo', 'ж': 'ʐ', 'з': 'z', 'и': 'i',
    'й': 'j', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
    'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
    'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'tɕ',
    'ш': 'ʂ', 'щ': 'ɕː', 'ъ': '', 'ы': 'ɨ', 'ь': 'ʲ',
    'э': 'e', 'ю': 'ju', 'я': 'ja'
}

# Диалектные особенности (севернорусские говоры)
DIALECT_FEATURES = {
    'оканье': True,  # сохранение 'о' в безударной позиции
    'цоканье': False,  # замена 'ч' на 'ц'
    'ёканье': True,  # 'е' -> 'ё' в некоторых позициях
}


def text_to_phonetic(text, dialect=True):
    """Преобразование текста в фонетическую транскрипцию с учетом диалекта"""
    text = text.lower().strip()
    phonetic = []

    i = 0
    while i < len(text):
        char = text[i]

        # Двухбуквенные сочетания
        if i < len(text) - 1:
            two = text[i:i + 2]
            if two == 'тс':
                phonetic.append('ts')
                i += 2
                continue
            elif two == 'дж':
                phonetic.append('dʐ')
                i += 2
                continue
            elif two == 'ого' and i < len(text) - 3 and text[i:i + 3] == 'ого':
                phonetic.append('ovo')
                i += 3
                continue
            elif two == 'его' and i < len(text) - 3 and text[i:i + 3] == 'его':
                phonetic.append('evo')
                i += 3
                continue

        # Диалектные замены
        if dialect:
            if DIALECT_FEATURES['цоканье'] and char == 'ч':
                phonetic.append('ts')
                i += 1
                continue
            if DIALECT_FEATURES['ёканье'] and char == 'е' and i > 0 and text[i - 1] not in 'аеёиоуыэюя':
                phonetic.append('jo')
                i += 1
                continue

        # Обычные буквы
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

    # Оглушение звонких на конце слов
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

    # Проверка формата
    allowed = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm', 'mpeg', 'mpga', 'oga', 'm4a', 'opus'}
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in allowed:
        return jsonify({'error': f'Формат не поддерживается. Разрешены: {", ".join(allowed)}'}), 400

    try:
        # Сохраняем загруженный файл во временную папку
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # В демо-режиме (без установленного Whisper/Vosk) используем тестовые данные
        # В реальном проекте здесь был бы вызов API распознавания

        # Имитация распознавания (замените на реальное API при необходимости)
        import hashlib
        hash_val = int(hashlib.md5(open(tmp_path, 'rb').read()).hexdigest()[:8], 16)

        # Демо-фразы для диалектной речи
        demo_phrases = [
            ("Орать пора, земля поспела", "a'ratʲ pa'ra zʲem'lʲa pa'spʲela"),
            ("Кочет кричит на заре", "'kotɕet krʲi'tɕit na za'rʲe"),
            ("Кушак повяжи потуже", "ku'ʂak pa'vʲaʐɨ pa'tuʐe"),
            ("Больно хорошо нынче", "'bolʲnə xəra'ʂo 'nɨnʲtɕe"),
            ("Поди сюды, родимый", "pa'dʲi sʲu'dɨ ra'dʲimɨj"),
        ]

        idx = hash_val % len(demo_phrases)
        text = demo_phrases[idx][0]

        # Если бы было реальное распознавание:
        # text = transcribe_with_whisper(tmp_path)

        phonetic = text_to_phonetic(text, dialect=True)

        # Удаляем временный файл
        os.unlink(tmp_path)

        return jsonify({
            'orthographic': text,
            'phonetic': phonetic
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# force redeploy


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
