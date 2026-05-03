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


# ==================== ВЫРАВНИВАНИЕ ====================

@app.route('/align', methods=['POST'])
def align_manuscripts():
    """Выравнивание с классификацией разночтений"""
    data = request.json
    manuscripts = data['manuscripts']
    main_index = int(data.get('main_index', 0))

    if main_index >= len(manuscripts):
        return jsonify({'error': 'Неверный индекс'}), 400

    # Слова главного списка
    main_words = manuscripts[main_index]['words']

    # Функции нормализации и сравнения
    def normalize(word):
        if not word or word == '—':
            return ''
        w = word.lower()
        for ch in '·⸱※∽⁘:.,;!?':
            w = w.replace(ch, '')
        for ch in '҃҄҅҆':
            w = w.replace(ch, '')
        w = w.rstrip('ъь')
        reps = {'ѣ': 'е', 'і': 'и', 'ї': 'и', 'ѵ': 'и', 'ѡ': 'о', 'ꙩ': 'о',
                'ѹ': 'у', 'ꙋ': 'у', 'ѧ': 'я', 'ꙗ': 'я', 'ѳ': 'ф', 'ѕ': 'з',
                'ꙁ': 'з', 'ꙑ': 'ы'}
        for old, new in reps.items():
            w = w.replace(old, new)
        return w

    def similarity(a, b):
        if a == b: return 10
        na, nb = normalize(a), normalize(b)
        if not na and not nb: return 0
        if na == nb: return 8
        anchors = {'и', 'же', 'въ', 'не', 'на', 'къ', 'съ', 'от', 'за', 'яко'}
        if na in anchors and nb in anchors: return 6
        max_len = max(len(na), len(nb))
        if max_len == 0: return 0
        matches = sum(1 for x, y in zip(na, nb) if x == y)
        ratio = matches / max_len
        if ratio > 0.8: return 6
        if ratio > 0.6: return 3
        if ratio > 0.4: return 1
        return -2

    def nw(seq1, seq2, gap=-5):
        n, m = len(seq1), len(seq2)
        dp = [[0.0]*(m+1) for _ in range(n+1)]
        tr = [['']*(m+1) for _ in range(n+1)]
        for i in range(1, n+1): dp[i][0] = i*gap; tr[i][0] = 'u'
        for j in range(1, m+1): dp[0][j] = j*gap; tr[0][j] = 'l'
        for i in range(1, n+1):
            for j in range(1, m+1):
                d = dp[i-1][j-1] + similarity(seq1[i-1], seq2[j-1])
                u = dp[i-1][j] + gap
                l = dp[i][j-1] + gap
                if d >= u and d >= l: dp[i][j]=d; tr[i][j]='d'
                elif u >= l: dp[i][j]=u; tr[i][j]='u'
                else: dp[i][j]=l; tr[i][j]='l'
        a1, a2 = [], []
        i, j = n, m
        while i>0 or j>0:
            if i>0 and j>0 and tr[i][j]=='d':
                a1.insert(0, seq1[i-1]); a2.insert(0, seq2[j-1]); i-=1; j-=1
            elif i>0 and tr[i][j]=='u':
                a1.insert(0, seq1[i-1]); a2.insert(0, '—'); i-=1
            else:
                a1.insert(0, '—'); a2.insert(0, seq2[j-1] if j>0 else '—')
                if j>0: j-=1
        return a1, a2

    # Классификация разночтений
    def detect_type(main_word, other_word):
        if main_word == '—': return 'insertion'
        if other_word == '—': return 'deletion'
        if main_word == other_word: return 'match'

        nm = normalize(main_word)
        no = normalize(other_word)
        if nm == no: return 'match'

        # Графические пары
        graphic_pairs = [('ѹ','у'),('ꙋ','у'),('оу','у'),('ѣ','е'),('ꙗ','я'),
                         ('ѧ','я'),('і','и'),('ї','и'),('ѵ','и'),('ѡ','о'),
                         ('ꙩ','о'),('ѳ','ф'),('ѕ','з'),('ꙁ','з')]
        tm, to_ = nm, no
        for old, new in graphic_pairs:
            tm = tm.replace(old, new)
            to_ = to_.replace(old, new)
        if tm == to_: return 'graphic'

        # Морфологические
        pref = 0
        for i in range(min(len(tm), len(to_))):
            if tm[i] == to_[i]: pref += 1
            else: break
        if pref >= 3:
            s1, s2 = tm[pref:], to_[pref:]
            if len(s1) <= 3 and len(s2) <= 3: return 'morph'

        # Фонетические
        if len(nm) == len(no) and len(nm) > 2:
            diff = sum(1 for a, b in zip(nm, no) if a != b)
            if diff <= 2: return 'phonetic'

        return 'lexical'

    # Выравниваем все списки попарно с главным
    other_names = []
    all_aligned = []

    for i, ms in enumerate(manuscripts):
        if i == main_index:
            continue
        _, aligned_other = nw(main_words, ms['words'])
        all_aligned.append(aligned_other)
        other_names.append(ms['name'])

    # Строим общую таблицу
    alignment_table = []
    diff_table = []
    stats = {'total':0,'match':0,'graphic':0,'phonetic':0,'morph':0,'lexical':0,'insertion':0,'deletion':0}

    max_len = max(len(a) for a in all_aligned)
    max_len = max(max_len, len(main_words))

    for i in range(max_len):
        mw = main_words[i] if i < len(main_words) else '—'
        row = [mw]
        types = []
        for ao in all_aligned:
            ow = ao[i] if i < len(ao) else '—'
            row.append(ow)
            dt = detect_type(mw, ow)
            types.append(dt)
            stats['total'] += 1
            if dt in stats: stats[dt] += 1
        alignment_table.append(row)
        diff_table.append(types)

    if stats['total'] > 0:
        stats['match_percent'] = round(stats['match']/stats['total']*100, 1)
    else:
        stats['match_percent'] = 0

    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        'alignment': alignment_table,
        'manuscript_names': [manuscripts[main_index]['name']] + other_names,
        'diff_types': diff_table,
        'statistics': stats
    }

    return jsonify({
        'session_id': session_id,
        'alignment': alignment_table,
        'manuscript_names': [manuscripts[main_index]['name']] + other_names,
        'diff_types': diff_table,
        'statistics': stats
    })


@app.route('/export_tei', methods=['POST'])
def export_tei():
    data = request.json or {}
    session_id = data.get('session_id')
    if session_id and session_id in sessions:
        ad = sessions[session_id]
    elif data.get('data'):
        ad = data['data']
    else:
        return jsonify({'error': 'Нет данных'}), 400

    names = ad.get('manuscript_names', ['Главный'])
    al = ad.get('alignment', [])

    xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml += '<TEI xmlns="http://www.tei-c.org/ns/1.0">\n'
    xml += '  <teiHeader>\n    <fileDesc>\n      <titleStmt>\n'
    xml += '        <title>Параллельный корпус</title>\n'
    xml += '      </titleStmt>\n      <sourceDesc>\n        <listWit>\n'
    for i, name in enumerate(names):
        wid = 'main' if i == 0 else f'wit{i}'
        xml += f'          <witness xml:id="{wid}">{name}</witness>\n'
    xml += '        </listWit>\n      </sourceDesc>\n    </fileDesc>\n'
    xml += '  </teiHeader>\n  <text>\n    <body>\n      <div type="alignment">\n'
    for i, row in enumerate(al):
        xml += f'        <app n="{i+1}">\n'
        xml += f'          <rdg wit="#main">{row[0] if row[0] else "—"}</rdg>\n'
        for j in range(1, len(row)):
            wid = f'wit{j}'
            xml += f'          <rdg wit="#{wid}">{row[j] if row[j] else "—"}</rdg>\n'
        xml += '        </app>\n'
    xml += '      </div>\n    </body>\n  </text>\n</TEI>'

    tmp = os.path.join('/tmp', f'export_{uuid.uuid4().hex[:8]}.xml')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(xml)
    return send_file(tmp, mimetype='application/xml', as_attachment=True, download_name='aligned_corpus.xml')


# ==================== RUN ====================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
