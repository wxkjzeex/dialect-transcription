import os
import tempfile
import uuid
import subprocess
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app)

# ==================== СЕССИИ ====================
sessions = {}

# Максимальный размер запроса - 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ==================== TEI ПАРСИНГ (оптимизированный) ====================

def parse_tei_fast(xml_content: str) -> Dict:
    """Быстрый парсинг TEI XML - только нужные теги"""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        return {'error': f'Ошибка парсинга XML: {str(e)}'}
    
    words = []
    sheets = []
    current_sheet = None
    
    # Только нужные теги для ускорения
    target_tags = {'w', 'pc', 'milestone'}
    
    for el in root.iter():
        tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
        
        if tag not in target_tags:
            continue
        
        if tag == 'milestone' and el.get('unit') == 'sheet':
            n = el.get('n')
            if n:
                try:
                    current_sheet = int(n)
                    sheets.append({
                        'index': len(words),
                        'sheet': current_sheet
                    })
                except ValueError:
                    pass
        
        elif tag in ['w', 'pc']:
            text = ''.join(el.itertext()).strip()
            if text and len(text) > 0 and text not in {'·', '⸱', '※', '∽', ':', '.', '⁘'}:
                words.append(text)
    
    if not words:
        return {'error': 'Не найдено слов в XML'}
    
    full_text = ' '.join(words)
    
    return {
        'words': words,
        'full_text': full_text,
        'word_count': len(words),
        'sheets': sheets
    }


def filter_by_sheets(data: Dict, start_sheet: int, end_sheet: int) -> List[str]:
    """Фильтрация слов по номерам листов"""
    if not start_sheet or not end_sheet:
        return data.get('words', [])
    
    words = data.get('words', [])
    sheets = data.get('sheets', [])
    
    if not sheets:
        return words
    
    start_idx = 0
    end_idx = len(words)
    
    for sheet in sheets:
        if sheet['sheet'] == start_sheet:
            start_idx = sheet['index']
        if sheet['sheet'] == end_sheet + 1:
            end_idx = sheet['index']
            break
    
    return words[start_idx:end_idx]


# ==================== НОРМАЛИЗАЦИЯ ====================

# Предварительно скомпилированные таблицы замен
REPLACEMENTS = str.maketrans({
    'ѣ': 'е', 'і': 'и', 'ї': 'и', 'ѵ': 'и',
    'ѡ': 'о', 'ꙩ': 'о', 'ѹ': 'у', 'ꙋ': 'у',
    'ѧ': 'я', 'ꙗ': 'я', 'ѳ': 'ф', 'ѕ': 'з',
    'ꙁ': 'з', 'ꙑ': 'ы', 'ѯ': 'к', 'ѱ': 'п'
})

PUNCTUATION = str.maketrans('', '', '·⸱※∽⁘:.,;!?\u0300-\u036f')
DIACRITICS = str.maketrans('', '', '҃҄҅҆')


def normalize_word(word: str) -> str:
    """Быстрая нормализация слова"""
    if not word or word == '—':
        return ''
    
    w = word.lower()
    w = w.translate(PUNCTUATION)
    w = w.translate(DIACRITICS)
    w = w.rstrip('ъь')
    w = w.translate(REPLACEMENTS)
    
    return w


# ==================== КЭШ СХОДСТВА ====================
_similarity_cache = {}


def word_similarity(w1: str, w2: str) -> float:
    """Оценка сходства двух слов с кэшированием"""
    cache_key = (w1, w2)
    if cache_key in _similarity_cache:
        return _similarity_cache[cache_key]
    
    if w1 == w2:
        _similarity_cache[cache_key] = 10.0
        return 10.0
    
    n1 = normalize_word(w1)
    n2 = normalize_word(w2)
    
    if not n1 and not n2:
        result = 0.0
    elif n1 == n2:
        result = 8.0
    elif len(n1) == 1 or len(n2) == 1:
        # Служебные слова
        anchors = {'и', 'в', 'с', 'к', 'о', 'у', 'а', 'н'}
        if n1 in anchors and n2 in anchors:
            result = 8.0
        else:
            result = -2.0
    else:
        # Упрощенное сравнение
        max_len = max(len(n1), len(n2))
        if max_len == 0:
            result = 0.0
        else:
            matches = sum(1 for a, b in zip(n1, n2) if a == b)
            ratio = matches / max_len
            
            if ratio > 0.8:
                result = 6.0
            elif ratio > 0.6:
                result = 3.0
            elif ratio > 0.4:
                result = 1.0
            else:
                result = -2.0
    
    _similarity_cache[cache_key] = result
    return result


# ==================== АЛГОРИТМЫ ВЫРАВНИВАНИЯ ====================

def needleman_wunsch(seq1: List[str], seq2: List[str], gap_penalty: float = -5.0) -> Tuple[List[str], List[str]]:
    """Оптимизированный алгоритм Нидлмана-Вунша"""
    n, m = len(seq1), len(seq2)
    
    # Ограничение на размер для скорости
    if n > 2000:
        n = 2000
        seq1 = seq1[:2000]
    if m > 2000:
        m = 2000
        seq2 = seq2[:2000]
    
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
            scores = {
                'd': dp[i-1][j-1] + match_score,
                'u': dp[i-1][j] + gap_penalty,
                'l': dp[i][j-1] + gap_penalty
            }
            best = max(scores, key=scores.get)
            dp[i][j] = scores[best]
            trace[i][j] = best
    
    aligned1, aligned2 = [], []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 'd':
            aligned1.append(seq1[i-1])
            aligned2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and trace[i][j] == 'u':
            aligned1.append(seq1[i-1])
            aligned2.append('—')
            i -= 1
        else:
            aligned1.append('—')
            aligned2.append(seq2[j-1] if j > 0 else '—')
            if j > 0:
                j -= 1
    
    return list(reversed(aligned1)), list(reversed(aligned2))


def multiple_alignment(main_words: List[str], other_words_list: List[List[str]]) -> Dict:
    """Множественное выравнивание"""
    if len(main_words) > 2000:
        main_words = main_words[:2000]
    
    alignments = []
    
    for other_words in other_words_list:
        if len(other_words) > 2000:
            other_words = other_words[:2000]
        aligned_main, aligned_other = needleman_wunsch(main_words, other_words)
        alignments.append({'main': aligned_main, 'other': aligned_other})
    
    # Определяем позиции вставок
    insertion_positions = set()
    for al in alignments:
        for pos, word in enumerate(al['main']):
            if word == '—':
                insertion_positions.add(pos)
    
    sorted_positions = sorted(insertion_positions)
    
    # Расширяем главный список
    expanded_main = []
    main_idx = 0
    pos_idx = 0
    
    max_len = max(len(al['main']) for al in alignments) if alignments else len(main_words)
    
    for pos in range(max_len + len(sorted_positions) + 100):
        if pos_idx < len(sorted_positions) and sorted_positions[pos_idx] == pos:
            expanded_main.append('—')
            pos_idx += 1
        elif main_idx < len(main_words):
            expanded_main.append(main_words[main_idx])
            main_idx += 1
        else:
            break
    
    # Выравниваем остальные
    result_alignments = []
    for al in alignments:
        row = []
        other_idx = 0
        
        for main_word in expanded_main:
            if main_word == '—':
                if other_idx < len(al['other']) and al['other'][other_idx] != '—':
                    row.append(al['other'][other_idx])
                    other_idx += 1
                else:
                    row.append('—')
            else:
                if other_idx < len(al['other']):
                    row.append(al['other'][other_idx])
                    other_idx += 1
                else:
                    row.append('—')
        
        result_alignments.append(row)
    
    return {
        'main_words': expanded_main,
        'alignments': result_alignments
    }


# ==================== КЛАССИФИКАЦИЯ ====================

GRAPHIC_MAP = {}
for pair in [('ѹ', 'у'), ('ꙋ', 'у'), ('оу', 'у'), ('ѡ', 'о'), ('ꙩ', 'о'),
             ('ѣ', 'е'), ('ꙗ', 'я'), ('ѧ', 'я'), ('і', 'и'), ('ї', 'и'),
             ('ѵ', 'и'), ('ѳ', 'ф'), ('ѕ', 'з'), ('ꙁ', 'з'), ('ꙑ', 'ы')]:
    GRAPHIC_MAP[pair[0]] = pair[1]


def detect_diff_type(main_word: str, other_word: str) -> str:
    """Определение типа разночтения"""
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
    
    # Графические
    test_main = ''.join(GRAPHIC_MAP.get(c, c) for c in norm_main)
    test_other = ''.join(GRAPHIC_MAP.get(c, c) for c in norm_other)
    
    if test_main == test_other:
        return 'graphic'
    
    # Фонетические (упрощенно)
    if len(norm_main) == len(norm_other):
        diff_count = sum(1 for a, b in zip(norm_main, norm_other) if a != b)
        if diff_count <= 2 and len(norm_main) > 2:
            return 'phonetic'
    
    # Морфологические
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
    """Проверка работоспособности"""
    return jsonify({'status': 'ok', 'timestamp': time.time()})


@app.route('/upload_corpus', methods=['POST'])
def upload_corpus():
    """Загрузка и парсинг XML-TEI файлов"""
    if 'files' not in request.files:
        return jsonify({'error': 'Нет файлов'}), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'Нет файлов'}), 400
    
    result = []
    
    for file in files:
        if not file.filename or not file.filename.endswith('.xml'):
            continue
        
        try:
            # Используем ваш существующий parse_tei
            data = parse_tei(file)
            
            result.append({
                'name': file.filename,
                'words': data['words'],
                'full_text': data['fullText'],
                'word_count': len(data['words']),
                'sheets': []  # Если нужно, добавьте парсинг листов
            })
            
        except Exception as e:
            return jsonify({'error': f'Ошибка в {file.filename}: {str(e)}'}), 400
    
    if not result:
        return jsonify({'error': 'Нет подходящих файлов'}), 400
    
    return jsonify({'manuscripts': result})

def normalize_word(word):
    """Быстрая нормализация слова для сравнения"""
    if not word or word == '—':
        return ''
    
    w = word.lower()
    # Удаляем пунктуацию
    for ch in '·⸱※∽⁘:.,;!?':
        w = w.replace(ch, '')
    # Удаляем диакритику
    for ch in '҃҄҅҆':
        w = w.replace(ch, '')
    w = w.rstrip('ъь')
    
    # Замены букв
    replacements = {
        'ѣ': 'е', 'і': 'и', 'ї': 'и', 'ѵ': 'и',
        'ѡ': 'о', 'ꙩ': 'о', 'ѹ': 'у', 'ꙋ': 'у',
        'ѧ': 'я', 'ꙗ': 'я', 'ѳ': 'ф', 'ѕ': 'з',
        'ꙁ': 'з', 'ꙑ': 'ы'
    }
    for old, new in replacements.items():
        w = w.replace(old, new)
    
    return w


def word_similarity(w1, w2):
    """Оценка сходства двух слов (0-10)"""
    if w1 == w2:
        return 10.0
    
    n1 = normalize_word(w1)
    n2 = normalize_word(w2)
    
    if not n1 and not n2:
        return 0.0
    if n1 == n2:
        return 8.0
    
    # Служебные слова
    anchors = {'и', 'же', 'въ', 'не', 'на', 'къ', 'съ', 'от', 'за', 'яко'}
    if n1 in anchors and n2 in anchors:
        return 6.0
    
    # Сравнение по длине общей части
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


def needleman_wunsch(seq1, seq2, gap_penalty=-5.0):
    """Упрощенный алгоритм Нидлмана-Вунша"""
    n, m = len(seq1), len(seq2)
    
    # Ограничение размера
    if n > 1000:
        n = 1000
        seq1 = seq1[:1000]
    if m > 1000:
        m = 1000
        seq2 = seq2[:1000]
    
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
            scores = {
                'd': dp[i-1][j-1] + match_score,
                'u': dp[i-1][j] + gap_penalty,
                'l': dp[i][j-1] + gap_penalty
            }
            best = max(scores, key=scores.get)
            dp[i][j] = scores[best]
            trace[i][j] = best
    
    aligned1, aligned2 = [], []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 'd':
            aligned1.append(seq1[i-1])
            aligned2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and trace[i][j] == 'u':
            aligned1.append(seq1[i-1])
            aligned2.append('—')
            i -= 1
        else:
            aligned1.append('—')
            aligned2.append(seq2[j-1] if j > 0 else '—')
            if j > 0:
                j -= 1
    
    return list(reversed(aligned1)), list(reversed(aligned2))


def detect_diff_type(main_word, other_word):
    """Определение типа разночтения"""
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
    
    # Графические
    graphic_pairs = [
        ('ѹ', 'у'), ('ꙋ', 'у'), ('оу', 'у'),
        ('ѣ', 'е'), ('ꙗ', 'я'), ('ѧ', 'я'),
        ('і', 'и'), ('ї', 'и'), ('ѵ', 'и'),
        ('ѡ', 'о'), ('ꙩ', 'о'), ('ѳ', 'ф'),
        ('ѕ', 'з'), ('ꙁ', 'з')
    ]
    
    test_main = norm_main
    test_other = norm_other
    for old, new in graphic_pairs:
        test_main = test_main.replace(old, new)
        test_other = test_other.replace(old, new)
    
    if test_main == test_other:
        return 'graphic'
    
    # Морфологические
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
    
    # Фонетические (упрощенно)
    if len(norm_main) == len(norm_other) and len(norm_main) > 2:
        diff_count = sum(1 for a, b in zip(norm_main, norm_other) if a != b)
        if diff_count <= 2:
            return 'phonetic'
    
    return 'lexical'


@app.route('/align', methods=['POST'])
def align_manuscripts():
    """Выравнивание рукописей"""
    data = request.json
    
    if not data or 'manuscripts' not in data:
        return jsonify({'error': 'Нет данных'}), 400
    
    manuscripts = data['manuscripts']
    main_index = int(data.get('main_index', 0))
    
    if main_index >= len(manuscripts):
        return jsonify({'error': 'Неверный индекс главного списка'}), 400
    
    # Главный список
    main_ms = manuscripts[main_index]
    main_words = main_ms['words']
    
    if not main_words:
        return jsonify({'error': 'Не найдены слова в главном списке'}), 400
    
    # Ограничение размера
    if len(main_words) > 500:
        main_words = main_words[:500]
    
    # Остальные списки
    other_names = []
    all_aligned = []
    
    for i, ms in enumerate(manuscripts):
        if i == main_index:
            continue
        
        other_words = ms['words'][:500]  # Ограничение
        
        # Попарное выравнивание с главным
        aligned_main, aligned_other = needleman_wunsch(main_words, other_words)
        all_aligned.append(aligned_other)
        other_names.append(ms['name'])
    
    if not all_aligned:
        return jsonify({'error': 'Нет списков для сравнения'}), 400
    
    # Определяем позиции вставок в главном
    insertion_positions = set()
    for al_other in all_aligned:
        # Используем первое выравнивание как эталон для позиций
        pass
    
    # Строим общее выравнивание (упрощенно)
    alignment_table = []
    diff_types_table = []
    
    max_len = max(len(al) for al in all_aligned)
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
        stats['match_percent'] = round(stats['match'] / stats['total'] * 100, 1)
    else:
        stats['match_percent'] = 0
    
    # Сохраняем сессию
    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        'alignment': alignment_table,
        'manuscript_names': [main_ms['name']] + other_names,
        'diff_types': diff_types_table,
        'statistics': stats
    }
    
    return jsonify({
        'session_id': session_id,
        'alignment': alignment_table,
        'manuscript_names': [main_ms['name']] + other_names,
        'diff_types': diff_types_table,
        'statistics': stats
    })


@app.route('/export_tei', methods=['POST'])
def export_tei():
    """Экспорт в TEI XML"""
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
    
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<TEI xmlns="http://www.tei-c.org/ns/1.0">\n'
    xml += '  <teiHeader>\n'
    xml += '    <fileDesc>\n'
    xml += '      <titleStmt>\n'
    xml += '        <title>Параллельный корпус: Притча о блудном сыне</title>\n'
    xml += '      </titleStmt>\n'
    xml += '      <sourceDesc>\n'
    xml += '        <listWit>\n'
    
    for i, name in enumerate(names):
        wid = 'main' if i == 0 else f'wit{i}'
        xml += f'          <witness xml:id="{wid}">{name}</witness>\n'
    
    xml += '        </listWit>\n'
    xml += '      </sourceDesc>\n'
    xml += '    </fileDesc>\n'
    xml += '  </teiHeader>\n'
    xml += '  <text>\n'
    xml += '    <body>\n'
    xml += '      <div type="alignment">\n'
    
    for i, row in enumerate(alignment):
        xml += f'        <app n="{i+1}">\n'
        xml += f'          <rdg wit="#main">{row[0] if row[0] else "—"}</rdg>\n'
        for j in range(1, len(row)):
            wid = f'wit{j}'
            xml += f'          <rdg wit="#{wid}">{row[j] if row[j] else "—"}</rdg>\n'
        xml += '        </app>\n'
    
    xml += '      </div>\n'
    xml += '    </body>\n'
    xml += '  </text>\n'
    xml += '</TEI>'
    
    # Сохраняем во временный файл
    temp_file = os.path.join('/tmp', f'export_{uuid.uuid4().hex[:8]}.xml')
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(xml)
    
    return send_file(
        temp_file,
        mimetype='application/xml',
        as_attachment=True,
        download_name='aligned_corpus.xml'
    )



# ==================== ЗАПУСК ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
