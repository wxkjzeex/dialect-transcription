import os
import tempfile
import uuid
import json
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional

app = Flask(__name__)
CORS(app)

# ==================== КОНФИГУРАЦИЯ ====================
UPLOAD_FOLDER = tempfile.mkdtemp()
SESSIONS = {}

# ==================== TEI ПАРСИНГ ====================

def parse_tei(xml_content: str) -> Dict:
    """Парсинг TEI XML с извлечением слов и листов"""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        # Пробуем как файл
        tree = ET.parse(xml_content)
        root = tree.getroot()
    
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    words = []
    sheets = []
    current_sheet = None
    
    # Ищем все элементы в правильном порядке
    for el in root.iter():
        # Убираем namespace
        tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
        
        # Отслеживаем листы
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
        
        # Извлекаем слова
        if tag in ['w', 'pc']:
            text = ''.join(el.itertext()).strip()
            if text and text not in ['·', '⸱', '※', '∽', ':', '.', '⁘']:
                words.append(text)
    
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
        return data['words']
    
    words = data['words']
    sheets = data.get('sheets', [])
    
    start_idx = 0
    end_idx = len(words)
    
    for sheet in sheets:
        if sheet['sheet'] == start_sheet:
            start_idx = sheet['index']
        if sheet['sheet'] == end_sheet + 1:
            end_idx = sheet['index']
            break
    
    return words[start_idx:end_idx]


# ==================== АЛГОРИТМЫ ВЫРАВНИВАНИЯ ====================

def normalize_word(word: str) -> str:
    """Нормализация слова для сравнения"""
    if not word or word == '—':
        return ''
    
    # Приводим к нижнему регистру
    w = word.lower()
    
    # Удаляем пунктуацию
    for ch in '·⸱※∽⁘:.,;!?':
        w = w.replace(ch, '')
    
    # Удаляем диакритику
    for ch in '҃҄҅҆':
        w = w.replace(ch, '')
    
    # Удаляем конечные ъ, ь
    w = w.rstrip('ъь')
    
    # Замены для церковнославянского
    replacements = {
        'ѣ': 'е', 'і': 'и', 'ї': 'и', 'ѵ': 'и',
        'ѡ': 'о', 'ꙩ': 'о', 'ѹ': 'у', 'ꙋ': 'у',
        'ѧ': 'я', 'ꙗ': 'я', 'ѳ': 'ф', 'ѯ': 'кс',
        'ѱ': 'пс', 'ѕ': 'з', 'ꙁ': 'з'
    }
    for old, new in replacements.items():
        w = w.replace(old, new)
    
    return w


def levenshtein_distance(s1: str, s2: str) -> int:
    """Расстояние Левенштейна"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    
    return dp[m][n]


def word_similarity(w1: str, w2: str) -> float:
    """Оценка сходства двух слов (0-10)"""
    if w1 == w2:
        return 10.0
    
    n1 = normalize_word(w1)
    n2 = normalize_word(w2)
    
    if not n1 and not n2:
        return 0.0
    if n1 == n2:
        return 8.0
    
    # Служебные слова-якоря
    anchors = {'и', 'же', 'въ', 'не', 'на', 'къ', 'съ', 'от', 'за', 'яко',
               'но', 'да', 'то', 'се', 'ли', 'бо', 'убо', 'аще'}
    if n1 in anchors and n2 in anchors:
        return 8.0
    
    # Сравнение по Левенштейну
    dist = levenshtein_distance(n1, n2)
    max_len = max(len(n1), len(n2))
    
    if max_len == 0:
        return 0.0
    
    ratio = dist / max_len
    
    if ratio < 0.2:
        return 6.0
    elif ratio < 0.4:
        return 3.0
    elif ratio < 0.6:
        return 1.0
    
    return -2.0


def needleman_wunsch(seq1: List[str], seq2: List[str], gap_penalty: float = -5.0) -> Tuple[List[str], List[str]]:
    """Алгоритм Нидлмана-Вунша для выравнивания двух последовательностей"""
    n, m = len(seq1), len(seq2)
    
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]
    
    # Инициализация
    for i in range(1, n + 1):
        dp[i][0] = i * gap_penalty
        trace[i][0] = 'up'
    
    for j in range(1, m + 1):
        dp[0][j] = j * gap_penalty
        trace[0][j] = 'left'
    
    # Заполнение матрицы
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i-1][j-1] + word_similarity(seq1[i-1], seq2[j-1])
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            
            max_val = max(match, delete, insert)
            dp[i][j] = max_val
            
            if max_val == match:
                trace[i][j] = 'diag'
            elif max_val == delete:
                trace[i][j] = 'up'
            else:
                trace[i][j] = 'left'
    
    # Обратный ход
    aligned1, aligned2 = [], []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 'diag':
            aligned1.insert(0, seq1[i-1])
            aligned2.insert(0, seq2[j-1])
            i -= 1
            j -= 1
        elif i > 0 and trace[i][j] == 'up':
            aligned1.insert(0, seq1[i-1])
            aligned2.insert(0, '—')
            i -= 1
        else:
            aligned1.insert(0, '—')
            aligned2.insert(0, seq2[j-1])
            j -= 1
    
    return aligned1, aligned2


def multiple_alignment(main_words: List[str], other_words_list: List[List[str]]) -> Dict:
    """Множественное выравнивание относительно главного списка"""
    alignments = []
    all_aligned_others = []
    
    for other_words in other_words_list:
        aligned_main, aligned_other = needleman_wunsch(main_words, other_words)
        alignments.append({
            'aligned_main': aligned_main,
            'aligned_other': aligned_other
        })
        all_aligned_others.append(aligned_other)
    
    # Определяем позиции для вставок в главном списке
    insertion_positions = set()
    for aligned_main, _ in alignments:
        for pos, word in enumerate(aligned_main):
            if word == '—':
                insertion_positions.add(pos)
    
    # Сортируем позиции
    sorted_positions = sorted(insertion_positions)
    
    # Строим расширенный главный список
    expanded_main = []
    main_idx = 0
    pos_set_idx = 0
    
    max_len = max(len(aligned_main) for aligned_main, _ in alignments)
    
    for pos in range(max_len + len(sorted_positions)):
        if pos_set_idx < len(sorted_positions) and sorted_positions[pos_set_idx] == pos:
            expanded_main.append('—')
            pos_set_idx += 1
        elif main_idx < len(main_words):
            expanded_main.append(main_words[main_idx])
            main_idx += 1
        else:
            expanded_main.append('—')
    
    # Выравниваем остальные списки
    result_alignments = []
    for _, aligned_other in alignments:
        result_row = []
        other_idx = 0
        
        for main_word in expanded_main:
            if main_word == '—':
                if other_idx < len(aligned_other) and aligned_other[other_idx] != '—':
                    result_row.append(aligned_other[other_idx])
                    other_idx += 1
                else:
                    result_row.append('—')
            else:
                if other_idx < len(aligned_other):
                    result_row.append(aligned_other[other_idx] if other_idx < len(aligned_other) else '—')
                    other_idx += 1
                else:
                    result_row.append('—')
        
        result_alignments.append(result_row)
    
    return {
        'main_words': expanded_main,
        'alignments': result_alignments
    }


# ==================== КЛАССИФИКАЦИЯ РАЗНОЧТЕНИЙ ====================

GRAPHIC_PAIRS = [
    ('ѹ', 'у'), ('ꙋ', 'у'), ('оу', 'у'),
    ('ѡ', 'о'), ('ꙩ', 'о'),
    ('ѣ', 'е'), ('ꙗ', 'я'), ('ѧ', 'я'),
    ('і', 'и'), ('ї', 'и'), ('ѵ', 'и'),
    ('ѳ', 'ф'), ('ѯ', 'кс'), ('ѱ', 'пс'),
    ('ꙁ', 'з'), ('ѕ', 'з'), ('ꙑ', 'ы')
]

PHONETIC_PATTERNS = [
    ('оро', 'ра'), ('оло', 'ла'), ('ере', 'ре'), ('ело', 'ле'),
    ('жд', 'ж'), ('щ', 'ч'), ('ц', 'ч'), ('з', 'ж'), ('с', 'ш')
]

MORPH_ENDINGS = [
    ('ъ', ''), ('ь', ''),
    ('ого', 'ова'), ('его', 'ева'),
    ('ыя', 'ой'), ('ия', 'ей'),
    ('ти', 'ть'), ('щи', 'чь')
]


def detect_diff_type(main_word: str, other_word: str) -> str:
    """Определение типа разночтения между двумя словами"""
    if not main_word or not other_word:
        return 'deletion' if main_word == '—' else 'insertion'
    
    if main_word == '—':
        return 'insertion'
    if other_word == '—':
        return 'deletion'
    
    if main_word == other_word:
        return 'match'
    
    # Нормализуем
    norm_main = normalize_word(main_word)
    norm_other = normalize_word(other_word)
    
    if norm_main == norm_other:
        return 'match'
    
    # Проверяем графические разночтения
    test_main = norm_main
    test_other = norm_other
    for old, new in GRAPHIC_PAIRS:
        test_main = test_main.replace(old, new)
        test_other = test_other.replace(old, new)
    
    if test_main == test_other:
        return 'graphic'
    
    # Проверяем фонетические
    for p1, p2 in PHONETIC_PATTERNS:
        if (p1 in norm_main and p2 in norm_other) or (p2 in norm_main and p1 in norm_other):
            return 'phonetic'
    
    # Проверяем морфологические (общая основа > 3 символов)
    common_prefix_len = 0
    for i in range(min(len(test_main), len(test_other))):
        if test_main[i] == test_other[i]:
            common_prefix_len += 1
        else:
            break
    
    if common_prefix_len >= 3:
        suffix1 = test_main[common_prefix_len:]
        suffix2 = test_other[common_prefix_len:]
        if len(suffix1) <= 3 and len(suffix2) <= 3:
            return 'morph'
    
    return 'lexical'


def calculate_statistics(alignment_data: Dict) -> Dict:
    """Подсчёт статистики разночтений"""
    stats = {
        'total': 0,
        'matches': 0,
        'graphic': 0,
        'phonetic': 0,
        'morph': 0,
        'lexical': 0,
        'insertion': 0,
        'deletion': 0
    }
    
    main_words = alignment_data['main_words']
    alignments = alignment_data['alignments']
    
    for i, main_word in enumerate(main_words):
        for other_row in alignments:
            if i < len(other_row):
                other_word = other_row[i]
                stats['total'] += 1
                
                diff_type = detect_diff_type(main_word, other_word)
                if diff_type in stats:
                    stats[diff_type] += 1
    
    if stats['total'] > 0:
        stats['match_percent'] = round(stats['matches'] / stats['total'] * 100, 1)
    else:
        stats['match_percent'] = 0
    
    return stats


def calculate_similarity_matrix(all_texts: List[List[str]]) -> Dict:
    """Расчёт матрицы попарного сходства списков"""
    n = len(all_texts)
    matrix = [[100.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            # Простое сравнение: процент одинаковых слов
            words1 = set(normalize_word(w) for w in all_texts[i])
            words2 = set(normalize_word(w) for w in all_texts[j])
            
            if not words1 and not words2:
                similarity = 100.0
            elif not words1 or not words2:
                similarity = 0.0
            else:
                intersection = words1 & words2
                union = words1 | words2
                similarity = round(len(intersection) / len(union) * 100, 1)
            
            matrix[i][j] = similarity
            matrix[j][i] = similarity
    
    return {
        'matrix': matrix,
        'labels': [f'M{i+1}' for i in range(n)]  # Заменяется на имена
    }


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


@app.route('/upload_corpus', methods=['POST'])
def upload_corpus():
    """Загрузка и парсинг TEI XML файлов"""
    if 'files' not in request.files:
        return jsonify({'error': 'Нет файлов'}), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'Нет файлов'}), 400
    
    result = []
    
    for file in files:
        if not file.filename.endswith('.xml'):
            continue
        
        content = file.read().decode('utf-8')
        parsed = parse_tei(content)
        
        result.append({
            'name': file.filename,
            'words': parsed['words'],
            'full_text': parsed['full_text'],
            'word_count': parsed['word_count'],
            'sheets': parsed['sheets']
        })
    
    if not result:
        return jsonify({'error': 'Нет подходящих файлов'}), 400
    
    return jsonify({'manuscripts': result})


@app.route('/align', methods=['POST'])
def align_manuscripts():
    """Выравнивание рукописей"""
    data = request.json
    
    if not data or 'manuscripts' not in data:
        return jsonify({'error': 'Нет данных'}), 400
    
    manuscripts = data['manuscripts']
    main_index = data.get('main_index', 0)
    folio_start = data.get('folio_start')
    folio_end = data.get('folio_end')
    
    # Фильтрация по листам для главного списка
    main_ms = manuscripts[main_index]
    if folio_start and folio_end:
        main_words = filter_by_sheets(
            {'words': main_ms['words'], 'sheets': main_ms.get('sheets', [])},
            int(folio_start),
            int(folio_end)
        )
    else:
        main_words = main_ms['words']
    
    # Остальные списки
    other_words_list = []
    other_names = []
    
    for i, ms in enumerate(manuscripts):
        if i != main_index:
            other_words_list.append(ms['words'])
            other_names.append(ms['name'])
    
    # Множественное выравнивание
    aligned = multiple_alignment(main_words, other_words_list)
    
    # Определение типов разночтений для каждой позиции
    diff_types = []
    for i, main_word in enumerate(aligned['main_words']):
        row_types = []
        for other_row in aligned['alignments']:
            if i < len(other_row):
                other_word = other_row[i]
                row_types.append(detect_diff_type(main_word, other_word))
            else:
                row_types.append('unknown')
        diff_types.append(row_types)
    
    # Статистика
    stats = calculate_statistics(aligned)
    
    # Имена списков
    all_names = [main_ms['name']] + other_names
    
    # Сходство
    sim_data = calculate_similarity_matrix([main_words] + other_words_list)
    sim_data['labels'] = all_names
    
    # Формирование таблицы выравнивания
    alignment_table = []
    for i, main_word in enumerate(aligned['main_words']):
        row = [main_word]
        for other_row in aligned['alignments']:
            row.append(other_row[i] if i < len(other_row) else '—')
        alignment_table.append(row)
    
    # Сессия
    session_id = str(uuid.uuid4())[:8]
    session_data = {
        'alignment': alignment_table,
        'manuscript_names': all_names,
        'diff_types': diff_types,
        'statistics': stats,
        'similarity_matrix': sim_data,
        'main_words': aligned['main_words'],
        'alignments': aligned['alignments']
    }
    SESSIONS[session_id] = session_data
    
    return jsonify({
        'session_id': session_id,
        'alignment': alignment_table,
        'manuscript_names': all_names,
        'diff_types': diff_types,
        'statistics': stats,
        'similarity_matrix': sim_data,
        'main_words': aligned['main_words'],
        'alignments': aligned['alignments']
    })


@app.route('/export_tei', methods=['POST'])
def export_tei():
    """Экспорт в TEI XML"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id and session_id in SESSIONS:
        alignment_data = SESSIONS[session_id]
    elif data.get('data'):
        alignment_data = data['data']
    else:
        return jsonify({'error': 'Нет данных для экспорта'}), 400
    
    # Формируем TEI XML
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<TEI xmlns="http://www.tei-c.org/ns/1.0">\n'
    xml += '  <teiHeader>\n'
    xml += '    <fileDesc>\n'
    xml += '      <titleStmt>\n'
    xml += '        <title>Параллельный корпус: Притча о блудном сыне</title>\n'
    xml += '      </titleStmt>\n'
    xml += '      <sourceDesc>\n'
    xml += '        <listWit>\n'
    
    for i, name in enumerate(alignment_data['manuscript_names']):
        wit_id = 'main' if i == 0 else f'wit{i}'
        xml += f'          <witness xml:id="{wit_id}">{name}</witness>\n'
    
    xml += '        </listWit>\n'
    xml += '      </sourceDesc>\n'
    xml += '    </fileDesc>\n'
    xml += '  </teiHeader>\n'
    xml += '  <text>\n'
    xml += '    <body>\n'
    xml += '      <div type="alignment">\n'
    
    # Аппарат разночтений
    xml += '        <app>\n'
    
    for i, row in enumerate(alignment_data['alignment']):
        xml += f'          <rdg wit="#main">{row[0]}</rdg>\n'
        for j in range(1, len(row)):
            wit_id = f'wit{j}'
            xml += f'          <rdg wit="#{wit_id}">{row[j]}</rdg>\n'
    
    xml += '        </app>\n'
    xml += '      </div>\n'
    xml += '    </body>\n'
    xml += '  </text>\n'
    xml += '</TEI>'
    
    # Сохраняем во временный файл и отправляем
    temp_file = tempfile.mktemp(suffix='.xml')
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(xml)
    
    return send_file(
        temp_file,
        mimetype='application/xml',
        as_attachment=True,
        download_name='aligned_corpus.xml'
    )


@app.route('/save_session', methods=['POST'])
def save_session():
    data = request.json
    session_id = str(uuid.uuid4())[:8]
    SESSIONS[session_id] = data
    return jsonify({'session_id': session_id})


@app.route('/load_session/<session_id>')
def load_session(session_id):
    if session_id in SESSIONS:
        return jsonify(SESSIONS[session_id])
    return jsonify({'error': 'Сессия не найдена'}), 404


# ==================== ЗАПУСК ====================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
