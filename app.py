import os
import tempfile
import uuid
import time
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

app = Flask(__name__)
CORS(app)

# ==================== СЕССИИ ====================
sessions = {}

# ==================== ФОНЕТИКА (из оригинального проекта) ====================

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
        voiced = {'b': 'p', 'v': 'f', 'g': 'k', 'd': 't',
                  'ʐ': 'ʂ', 'z': 's'}
        if phon and phon[-1] in voiced:
            phon[-1] = voiced[phon[-1]]

        result.append(''.join(phon))

    return ' '.join(result)


# ==================== TEI ПАРСИНГ ====================

def parse_tei(xml_file):
    """Парсинг XML-TEI (для старых маршрутов)"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    words = []

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


# ==================== АЛГОРИТМЫ ВЫРАВНИВАНИЯ ====================

# Замены для нормализации
REPLACEMENTS = str.maketrans({
    'ѣ': 'е', 'і': 'и', 'ї': 'и', 'ѵ': 'и',
    'ѡ': 'о', 'ꙩ': 'о', 'ѹ': 'у', 'ꙋ': 'у',
    'ѧ': 'я', 'ꙗ': 'я', 'ѳ': 'ф', 'ѕ': 'з',
    'ꙁ': 'з', 'ꙑ': 'ы'
})

PUNCTUATION = str.maketrans('', '', '·⸱※∽⁘:.,;!?')
DIACRITICS = str.maketrans('', '', '҃҄҅҆')


def normalize_word(word: str) -> str:
    """Нормализация слова для сравнения"""
    if not word or word == '—':
        return ''

    w = word.lower()
    w = w.translate(PUNCTUATION)
    w = w.translate(DIACRITICS)
    w = w.rstrip('ъь')
    w = w.translate(REPLACEMENTS)

    return w


def word_similarity(w1: str, w2: str) -> float:
    """Оценка сходства двух слов (от -2 до 10)"""
    if w1 == w2:
        return 10.0

    n1 = normalize_word(w1)
    n2 = normalize_word(w2)

    if not n1 and not n2:
        return 0.0
    if n1 == n2:
        return 8.0

    # Служебные слова-якоря
    anchors = {'и', 'же', 'въ', 'не', 'на', 'къ', 'съ', 'от', 'за', 'яко'}
    if n1 in anchors and n2 in anchors:
        return 6.0

    max_len = max(len(n1), len(n2))
    if max_len == 0:
        return 0.0

    matches = sum(1 for a, b in zip(n1, n2) if a == b)
    ratio = matches / max_len

    if ratio > 0.8:
        return 6.0
    elif ratio > 0.6:
        return 3.0
    elif ratio > 0.4:
        return 1.0

    return -2.0


def needleman_wunsch(seq1: List[str], seq2: List[str],
                     gap_penalty: float = -5.0) -> Tuple[List[str], List[str]]:
    """Алгоритм Нидлмана-Вунша для выравнивания двух списков слов"""
    n, m = len(seq1), len(seq2)

    # Ограничение размера для производительности
    if n > 500:
        n = 500
        seq1 = seq1[:500]
    if m > 500:
        m = 500
        seq2 = seq2[:500]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[''] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * gap_penalty
        trace[i][0] = 'u'

    for j in range(1, m + 1):
        dp[0][j] = j * gap_penalty
        trace[0][j] = 'l'

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = word_similarity(seq1[i-1], seq2[j-1])
            diag = dp[i-1][j-1] + match_score
            up = dp[i-1][j] + gap_penalty
            left = dp[i][j-1] + gap_penalty

            if diag >= up and diag >= left:
                dp[i][j] = diag
                trace[i][j] = 'd'
            elif up >= left:
                dp[i][j] = up
                trace[i][j] = 'u'
            else:
                dp[i][j] = left
                trace[i][j] = 'l'

    aligned1, aligned2 = [], []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 'd':
            aligned1.insert(0, seq1[i-1])
            aligned2.insert(0, seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and trace[i][j] == 'u':
            aligned1.insert(0, seq1[i-1])
            aligned2.insert(0, '—')
            i -= 1
        else:
            aligned1.insert(0, '—')
            aligned2.insert(0, seq2[j-1] if j > 0 else '—')
            if j > 0:
                j -= 1

    return aligned1, aligned2


# ==================== КЛАССИФИКАЦИЯ РАЗНОЧТЕНИЙ ====================

GRAPHIC_PAIRS = [
    ('ѹ', 'у'), ('ꙋ', 'у'), ('оу', 'у'),
    ('ѣ', 'е'), ('ꙗ', 'я'), ('ѧ', 'я'),
    ('і', 'и'), ('ї', 'и'), ('ѵ', 'и'),
    ('ѡ', 'о'), ('ꙩ', 'о'), ('ѳ', 'ф'),
    ('ѕ', 'з'), ('ꙁ', 'з')
]


def detect_diff_type(main_word: str, other_word: str) -> str:
    """Определение типа разночтения между словами"""
    if main_word == '—':
        return 'insertion'
    if other_word == '—':
        return 'deletion'
    if main_word == other_word:
        return 'match'

    norm_main = normalize_word(main_word)
    norm_other = normalize_word(other_word)

    if norm_main == norm_other:
        return 'match'

    # Графические: заменяем буквы и проверяем снова
    test_main = norm_main
    test_other = norm_other
    for old, new in GRAPHIC_PAIRS:
        test_main = test_main.replace(old, new)
        test_other = test_other.replace(old, new)

    if test_main == test_other:
        return 'graphic'

    # Морфологические: общая основа > 3, разные окончания ≤ 3
    prefix_len = 0
    for i in range(min(len(test_main), len(test_other))):
        if test_main[i] == test_other[i]:
            prefix_len += 1
        else:
            break

    if prefix_len >= 3:
        suffix1 = test_main[prefix_len:]
        suffix2 = test_other[prefix_len:]
        if len(suffix1) <= 3 and len(suffix2) <= 3:
            return 'morph'

    # Фонетические: одинаковая длина, разница в 1-2 буквы
    if len(norm_main) == len(norm_other) and len(norm_main) > 2:
        diff_count = sum(1 for a, b in zip(norm_main, norm_other) if a != b)
        if diff_count <= 2:
            return 'phonetic'

    return 'lexical'


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


@app.route('/health')
def health():
    """Проверка работоспособности сервера"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'sessions': len(sessions)
    })


# ==================== СТАРЫЕ МАРШРУТЫ (ТРАНСКРИПЦИЯ) ====================

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


# ==================== НОВЫЕ МАРШРУТЫ (КОРПУС) ====================

@app.route('/align', methods=['POST'])
def align_manuscripts():
    """Серверное выравнивание списков"""
    data = request.json

    if not data or 'manuscripts' not in data:
        return jsonify({'error': 'Нет данных'}), 400

    manuscripts = data['manuscripts']
    main_index = int(data.get('main_index', 0))

    if main_index >= len(manuscripts):
        return jsonify({'error': 'Неверный индекс главного списка'}), 400

    # Главный список
    main_ms = manuscripts[main_index]
    main_words = main_ms.get('words', [])

    if not main_words:
        return jsonify({'error': 'Не найдены слова в главном списке'}), 400

    # Ограничение размера
    if len(main_words) > 500:
        main_words = main_words[:500]

    # Попарное выравнивание с каждым другим списком
    other_names = []
    all_aligned = []

    start_time = time.time()

    for i, ms in enumerate(manuscripts):
        if i == main_index:
            continue

        other_words = ms.get('words', [])[:500]

        # Выравнивание
        aligned_main, aligned_other = needleman_wunsch(main_words, other_words)
        all_aligned.append(aligned_other)
        other_names.append(ms.get('name', f'Список {i+1}'))

    if not all_aligned:
        return jsonify({'error': 'Нет списков для сравнения'}), 400

    # Строим общую таблицу
    alignment_table = []
    diff_types_table = []

    max_len = max(len(a) for a in all_aligned)
    max_len = max(max_len, len(main_words))

    for i in range(max_len):
        main_word = main_words[i] if i < len(main_words) else '—'
        row = [main_word]
        types_row = []

        for al_other in all_aligned:
            other_word = al_other[i] if i < len(al_other) else '—'
            row.append(other_word)
            types_row.append(detect_diff_type(main_word, other_word))

        alignment_table.append(row)
        diff_types_table.append(types_row)

    # Статистика
    stats = {
        'total': 0,
        'match': 0,
        'graphic': 0,
        'phonetic': 0,
        'morph': 0,
        'lexical': 0,
        'insertion': 0,
        'deletion': 0
    }

    for types_row in diff_types_table:
        for dt in types_row:
            stats['total'] += 1
            if dt in stats:
                stats[dt] += 1

    if stats['total'] > 0:
        stats['matchPercent'] = round(stats['match'] / stats['total'] * 100, 1)
    else:
        stats['matchPercent'] = 0

    elapsed = round(time.time() - start_time, 2)

    # Сохраняем сессию
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        'alignment': alignment_table,
        'manuscript_names': [main_ms.get('name', 'Главный')] + other_names,
        'diff_types': diff_types_table,
        'statistics': stats
    }

    return jsonify({
        'session_id': session_id,
        'alignment': alignment_table,
        'manuscript_names': [main_ms.get('name', 'Главный')] + other_names,
        'diff_types': diff_types_table,
        'statistics': stats,
        'align_time': elapsed
    })


@app.route('/export_tei', methods=['POST'])
def export_tei():
    """Экспорт результатов в TEI XML"""
    data = request.json or {}
    session_id = data.get('session_id')

    if session_id and session_id in sessions:
        alignment_data = sessions[session_id]
    elif data.get('data'):
        alignment_data = data['data']
    else:
        return jsonify({'error': 'Нет данных для экспорта'}), 400

    names = alignment_data.get('manuscript_names', ['Главный'])
    alignment = alignment_data.get('alignment', [])

    if not alignment:
        return jsonify({'error': 'Нет данных выравнивания'}), 400

    # Генерируем TEI XML
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<TEI xmlns="http://www.tei-c.org/ns/1.0">',
        '  <teiHeader>',
        '    <fileDesc>',
        '      <titleStmt>',
        '        <title>Параллельный корпус: Притча о блудном сыне</title>',
        '      </titleStmt>',
        '      <sourceDesc>',
        '        <listWit>'
    ]

    for i, name in enumerate(names):
        wid = 'main' if i == 0 else f'wit{i}'
        xml_parts.append(f'          <witness xml:id="{wid}">{name}</witness>')

    xml_parts.extend([
        '        </listWit>',
        '      </sourceDesc>',
        '    </fileDesc>',
        '  </teiHeader>',
        '  <text>',
        '    <body>',
        '      <div type="alignment">'
    ])

    for i, row in enumerate(alignment):
        xml_parts.append(f'        <app n="{i+1}">')
        xml_parts.append(f'          <rdg wit="#main">{escape_xml(row[0])}</rdg>')
        for j in range(1, len(row)):
            wid = f'wit{j}'
            xml_parts.append(f'          <rdg wit="#{wid}">{escape_xml(row[j])}</rdg>')
        xml_parts.append('        </app>')

    xml_parts.extend([
        '      </div>',
        '    </body>',
        '  </text>',
        '</TEI>'
    ])

    xml_content = '\n'.join(xml_parts)

    # Сохраняем и отправляем
    temp_file = os.path.join('/tmp', f'export_{uuid.uuid4().hex[:8]}.xml')
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(xml_content)

    return send_file(
        temp_file,
        mimetype='application/xml',
        as_attachment=True,
        download_name='aligned_corpus.xml'
    )


def escape_xml(text: str) -> str:
    """Экранирование специальных символов XML"""
    if not text:
        return '—'
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


# ==================== ЗАПУСК ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
