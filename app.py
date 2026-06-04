from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image, ExifTags
import os
import io
import base64
import piexif
import json
import numbers

from db import get_db, init_db

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'heic'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Серіалізація EXIF-значень ──────────────────────────────────────────────────

def serialize_exif_value(val):
    if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
        try:
            return float(val)
        except Exception:
            return str(val)
    elif isinstance(val, bytes):
        try:
            s = val.decode('utf-8', errors='replace')
        except Exception:
            s = str(val)
        # Спроба перекодувати latin1→utf-8 для кирилиці
        if 'Ð' in s or 'Ñ' in s:
            try:
                s2 = val.decode('latin1').encode('latin1').decode('utf-8', errors='replace')
                if any('\u0400' <= c <= '\u04FF' for c in s2):
                    s = s2
            except Exception:
                pass
        return s
    elif isinstance(val, (list, tuple)):
        return [serialize_exif_value(v) for v in val]
    elif isinstance(val, dict):
        return {serialize_exif_value(k): serialize_exif_value(v) for k, v in val.items()}
    elif isinstance(val, numbers.Number):
        return val
    else:
        return str(val)


# ── Вилучення EXIF ────────────────────────────────────────────────────────────

def get_exif_data(image):
    exif_data = {}
    try:
        exif_raw = image._getexif()
        if exif_raw:
            for tag, value in exif_raw.items():
                decoded = ExifTags.TAGS.get(tag, str(tag))
                # Пропускаємо бінарні поля без сенсу для користувача
                if decoded in ('MakerNote', 'UserComment') and isinstance(value, bytes) and len(value) > 64:
                    continue
                exif_data[decoded] = serialize_exif_value(value)
    except Exception:
        pass
    return exif_data


# ── Розбивка метаданих на секції ──────────────────────────────────────────────

CAMERA_FIELDS = {
    'Make', 'Model', 'LensModel', 'LensMake', 'FocalLength',
    'FocalLengthIn35mmFilm', 'MaxApertureValue', 'ApertureValue',
    'FNumber', 'ExposureTime', 'ISOSpeedRatings', 'Flash',
    'MeteringMode', 'ExposureMode', 'ExposureProgram', 'WhiteBalance',
    'DigitalZoomRatio', 'SceneCaptureType',
}

TECHNICAL_FIELDS = {
    'Software', 'DateTime', 'DateTimeOriginal', 'DateTimeDigitized',
    'ImageDescription', 'Artist', 'Copyright', 'XResolution',
    'YResolution', 'ResolutionUnit', 'ColorSpace', 'Orientation',
    'ExifVersion', 'FlashPixVersion', 'ComponentsConfiguration',
    'CompressedBitsPerPixel', 'BrightnessValue', 'ExposureBiasValue',
    'SubjectDistance', 'LightSource', 'SensingMethod', 'FileSource',
    'SceneType', 'CFAPattern', 'CustomRendered', 'PixelXDimension',
    'PixelYDimension', 'SubjectDistanceRange',
}


def split_exif_sections(exif_data):
    """Розбиває плоский словник EXIF на секції: camera / technical / інше."""
    camera, technical, other = {}, {}, {}
    for k, v in exif_data.items():
        if k in CAMERA_FIELDS:
            camera[k] = v
        elif k in TECHNICAL_FIELDS:
            technical[k] = v
        else:
            other[k] = v
    return camera, technical, other


# ── GPS ───────────────────────────────────────────────────────────────────────

def get_gps_info(exif_data):
    gps_info = {}
    gps = exif_data.get('GPSInfo')
    if not gps:
        return gps_info

    def _to_float(x):
        if isinstance(x, (list, tuple)) and len(x) == 2:
            return x[0] / x[1] if x[1] != 0 else 0.0
        return float(x)

    def _convert(coord, ref):
        d, m, s = [_to_float(c) for c in coord]
        val = d + m / 60 + s / 3600
        if ref in ('S', 'W'):
            val = -val
        return round(val, 8)

    try:
        if 2 in gps and 1 in gps:
            gps_info['latitude_decimal'] = _convert(gps[2], gps[1])
        if 4 in gps and 3 in gps:
            gps_info['longitude_decimal'] = _convert(gps[4], gps[3])
        if 6 in gps:
            alt = gps[6]
            gps_info['GPSAltitude'] = round(_to_float(alt) if not isinstance(alt, (list, tuple)) else alt[0] / alt[1], 2)
    except Exception:
        pass

    return gps_info


# ── AI-евристика ──────────────────────────────────────────────────────────────

def ai_heuristics(exif_data, file_info=None):
    score = 0
    signals = []

    # 1. Відсутність EXIF (+25)
    if not exif_data:
        score += 25
        signals.append({'signal': 'Відсутність EXIF', 'description': 'EXIF-дані відсутні — характерно для AI-зображень', 'type': 'warning', 'weight': 25})

    # 2. AI-слово у Software (+40)
    software = str(exif_data.get('Software', '')).lower()
    if software and any(w in software for w in ('diffusion', 'midjourney', 'dalle', 'stable', 'gencraft', 'invoke', 'comfy')):
        score += 40
        signals.append({'signal': 'AI-інструмент у Software', 'description': str(exif_data.get('Software')), 'type': 'danger', 'weight': 40})

    # 3. XMP CreatorTool (+35)
    creator = exif_data.get('XMP:CreatorTool') or exif_data.get('CreatorTool', '')
    if creator and any(w in str(creator).lower() for w in ('ai', 'midjourney', 'diffusion', 'dalle')):
        score += 35
        signals.append({'signal': 'XMP CreatorTool — AI', 'description': str(creator), 'type': 'danger', 'weight': 35})

    # 4. Стандартні AI-розміри (+12)
    AI_SIZES = {(512, 512), (768, 768), (1024, 1024), (768, 1344), (1344, 768),
                (512, 768), (768, 512), (1024, 1792), (1792, 1024)}
    if file_info:
        w, h = file_info.get('width', 0), file_info.get('height', 0)
        if (w, h) in AI_SIZES:
            score += 12
            signals.append({'signal': 'Стандартний AI-розмір', 'description': f'{w}×{h} px', 'type': 'warning', 'weight': 12})

    # 5. Відсутні ключові поля камери (+18)
    if exif_data and all(not exif_data.get(k) for k in ('DateTime', 'Model', 'LensModel')):
        score += 18
        signals.append({'signal': 'Відсутні дані камери', 'description': 'Немає дати, моделі камери та об\'єктиву', 'type': 'warning', 'weight': 18})

    # Позитивні сигнали (знижують оцінку)
    if exif_data and any(exif_data.get(k) for k in ('Model', 'LensModel', 'DateTime')):
        score = max(0, score - 15)
        signals.append({'signal': 'Присутні метадані камери', 'description': 'Реальна фотографія зазвичай має ці дані', 'type': 'good', 'weight': -15})

    if file_info and file_info.get('format') in ('JPEG', 'TIFF'):
        score = max(0, score - 10)
        signals.append({'signal': f'Формат {file_info.get("format")}', 'description': 'Типовий формат для фотокамер', 'type': 'good', 'weight': -10})

    score = min(100, max(0, score))
    return {'score': score, 'signals': signals}


# ── IPTC ──────────────────────────────────────────────────────────────────────

def get_iptc_data(path):
    """Вилучає IPTC-метадані з файлу через iptcinfo3."""
    try:
        import iptcinfo3
        info = iptcinfo3.IPTCInfo(path, force=True)
        result = {}
        # Мапа кодів IPTC на людські назви
        IPTC_TAGS = {
            5: 'ObjectName', 7: 'EditStatus', 10: 'Urgency', 15: 'Category',
            20: 'SupplementalCategory', 22: 'FixtureIdentifier', 25: 'Keywords',
            30: 'ReleaseDate', 35: 'ReleaseTime', 40: 'SpecialInstructions',
            45: 'ReferenceService', 47: 'ReferenceDate', 50: 'ReferenceNumber',
            55: 'DateCreated', 60: 'TimeCreated', 62: 'DigitalCreationDate',
            65: 'DigitalCreationTime', 70: 'OriginatingProgram',
            75: 'ProgramVersion', 80: 'Byline', 85: 'BylineTitle',
            90: 'City', 92: 'SubLocation', 95: 'Province',
            100: 'CountryCode', 101: 'Country', 103: 'OriginalTransmissionRef',
            105: 'Headline', 110: 'Credit', 115: 'Source',
            116: 'CopyrightNotice', 118: 'Contact', 120: 'Caption',
            122: 'Writer',
        }
        raw = info._data if hasattr(info, '_data') else {}
        for code, value in raw.items():
            if not value:
                continue
            name = IPTC_TAGS.get(code, str(code))
            if isinstance(value, list):
                decoded = [v.decode('utf-8', errors='replace') if isinstance(v, bytes) else str(v) for v in value]
                result[name] = ', '.join(decoded) if len(decoded) > 1 else decoded[0]
            elif isinstance(value, bytes):
                result[name] = value.decode('utf-8', errors='replace')
            else:
                result[name] = str(value)
        return result if result else None
    except Exception:
        return None


# ── XMP ───────────────────────────────────────────────────────────────────────

def get_xmp_data(image):
    """Вилучає XMP-метадані з PIL Image.info."""
    try:
        xmp_str = None
        if 'XML:com.adobe.xmp' in image.info:
            xmp_str = image.info['XML:com.adobe.xmp']
        elif 'xmp' in image.info:
            xmp_str = image.info['xmp']
            if isinstance(xmp_str, bytes):
                xmp_str = xmp_str.decode('utf-8', errors='replace')
        if not xmp_str:
            return None

        import xml.etree.ElementTree as ET
        result = {}
        root = ET.fromstring(xmp_str)
        for elem in root.iter():
            tag = elem.tag
            # Прибираємо namespace
            if '}' in tag:
                tag = tag.split('}', 1)[1]
            if elem.text and elem.text.strip():
                result[tag] = elem.text.strip()
        return result if result else None
    except Exception:
        return None


# ── Thumbnail ─────────────────────────────────────────────────────────────────

def make_thumbnail(image_path):
    try:
        img = Image.open(image_path)
        img.thumbnail((256, 256))
        if img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            bg.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

STATIC_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'ico', 'svg', 'webp',
                     'css', 'js', 'woff', 'woff2', 'ttf', 'mp4', 'webm'}


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def static_assets(filename):
    """Роздає будь-який статичний файл з кореневої папки додатку."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext in STATIC_EXTENSIONS and not filename.startswith('api/'):
        return send_from_directory('.', filename)
    from flask import abort
    abort(404)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не передано'}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Файл не обрано'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Непідтримуваний тип файлу'}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    try:
        image = Image.open(path)
    except Exception as e:
        os.remove(path)
        return jsonify({'error': f'Не вдалося відкрити зображення: {e}'}), 400

    # Збираємо розміри ДО thumbnail
    file_info = {
        'width': image.width,
        'height': image.height,
        'format': image.format or path.rsplit('.', 1)[-1].upper(),
        'mode': image.mode,
        'megapixels': round(image.width * image.height / 1_000_000, 2),
    }

    exif_data = get_exif_data(image)
    gps = get_gps_info(exif_data)
    camera, technical, other_exif = split_exif_sections(exif_data)
    iptc_data = get_iptc_data(path)
    xmp_data = get_xmp_data(image)
    ai = ai_heuristics(exif_data, file_info)

    # Thumbnail окремо — не впливає на file_info
    thumb_b64 = make_thumbnail(path)

    metadata = {
        'exif': exif_data,
        'camera': camera,
        'technical': technical,
        'other': other_exif,
        'iptc': iptc_data,
        'xmp': xmp_data,
        'gps': gps,
        'file_info': file_info,
    }

    image_id = None
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            '''INSERT INTO images
               (filename, original_name, file_size, mime_type, width, height, metadata_json, ai_score, ai_analysis_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                filename,
                file.filename,
                os.path.getsize(path),
                file.mimetype,
                file_info['width'],
                file_info['height'],
                json.dumps(metadata, ensure_ascii=False),
                ai['score'],
                json.dumps(ai, ensure_ascii=False),
            )
        )
        conn.commit()
        image_id = c.lastrowid
        conn.close()
    except Exception as e:
        print('DB error:', e)

    return jsonify({
        'filename': filename,
        'file_size': os.path.getsize(path),
        'metadata': metadata,
        'ai_analysis': ai,
        'thumbnail': thumb_b64,
        'image_id': image_id,
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('''SELECT id, filename, original_name, file_size, mime_type,
                            width, height, upload_time, ai_score, ai_analysis_json, metadata_json
                     FROM images ORDER BY upload_time DESC LIMIT 100''')
        rows = c.fetchall()
        conn.close()
        images = []
        for row in rows:
            images.append({
                'id': row['id'],
                'name': row['original_name'],
                'filename': row['filename'],
                'file_size': row['file_size'],
                'size': row['file_size'],
                'mime_type': row['mime_type'],
                'width': row['width'],
                'height': row['height'],
                'upload_time': row['upload_time'],
                'ai_score': row['ai_score'],
                'ai_analysis': json.loads(row['ai_analysis_json']) if row['ai_analysis_json'] else None,
                'metadata': json.loads(row['metadata_json']) if row['metadata_json'] else None,
            })
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<int:image_id>', methods=['GET'])
def get_image(image_id):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('''SELECT id, filename, original_name, file_size, width, height,
                            upload_time, ai_score, ai_analysis_json, metadata_json
                     FROM images WHERE id = ?''', (image_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'Зображення не знайдено'}), 404
        return jsonify({
            'id': row['id'],
            'filename': row['filename'],
            'original_name': row['original_name'],
            'file_size': row['file_size'],
            'width': row['width'],
            'height': row['height'],
            'upload_time': row['upload_time'],
            'ai_analysis': json.loads(row['ai_analysis_json']) if row['ai_analysis_json'] else {},
            'metadata': json.loads(row['metadata_json']) if row['metadata_json'] else {},
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<int:image_id>/history', methods=['GET'])
def get_image_history(image_id):
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('''SELECT field_name, old_value, new_value, edit_time
                     FROM metadata_edits WHERE image_id = ?
                     ORDER BY edit_time DESC LIMIT 50''', (image_id,))
        rows = c.fetchall()
        conn.close()
        return jsonify({'history': [
            {'field': r['field_name'], 'old': r['old_value'], 'new': r['new_value'], 'time': r['edit_time']}
            for r in rows
        ]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<int:image_id>/edit', methods=['POST'])
def edit_exif(image_id):
    """Редагування EXIF-метаданих. Підтримує поля з 0th і Exif IFD."""
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT filename FROM images WHERE id = ?', (image_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return jsonify({'error': 'Зображення не знайдено'}), 404
        filename = row['filename']
        conn.close()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(path):
        return jsonify({'error': 'Файл не знайдено на диску'}), 404

    try:
        img = Image.open(path)
        if img.format != 'JPEG':
            return jsonify({'error': 'Редагування EXIF підтримується лише для JPEG'}), 400

        edits = request.json.get('edits', {})
        exif_bytes = img.info.get('exif', b'')
        exif_dict = piexif.load(exif_bytes) if exif_bytes else {'0th': {}, 'Exif': {}, 'GPS': {}, '1st': {}}

        # Збираємо старі значення перед змінами
        old_values = {}

        # Побудуємо зворотну мапу: назва тегу → (IFD, tag_id)
        tag_map = {}
        for ifd_name, ifd_tags in [('0th', piexif.ImageIFD), ('Exif', piexif.ExifIFD)]:
            for attr in dir(ifd_tags):
                if attr.startswith('_'):
                    continue
                tag_id = getattr(ifd_tags, attr)
                if isinstance(tag_id, int):
                    tag_map[attr] = (ifd_name, tag_id)

        for key, new_val in edits.items():
            # Очікуємо формат "exif.FieldName" або "technical.FieldName" тощо
            parts = key.split('.', 1)
            field = parts[1] if len(parts) == 2 else parts[0]

            if field not in tag_map:
                continue

            ifd_name, tag_id = tag_map[field]

            # Зберігаємо старе значення
            old_raw = exif_dict.get(ifd_name, {}).get(tag_id)
            if old_raw is not None:
                old_values[field] = old_raw.decode('utf-8', errors='replace') if isinstance(old_raw, bytes) else str(old_raw)
            else:
                old_values[field] = None

            # Записуємо нове (ASCII)
            try:
                exif_dict.setdefault(ifd_name, {})[tag_id] = new_val.encode('utf-8', errors='replace')
            except Exception:
                continue

        new_exif_bytes = piexif.dump(exif_dict)
        img.save(path, exif=new_exif_bytes)

        # Зберігаємо в БД з правильним old_value
        try:
            conn = get_db()
            c = conn.cursor()
            for key, new_val in edits.items():
                parts = key.split('.', 1)
                field = parts[1] if len(parts) == 2 else parts[0]
                c.execute(
                    '''INSERT INTO metadata_edits (image_id, field_name, old_value, new_value)
                       VALUES (?, ?, ?, ?)''',
                    (image_id, field, old_values.get(field), new_val)
                )
            conn.commit()
            conn.close()
        except Exception as e:
            print('DB edit error:', e)

        # Повертаємо оновлені метадані
        img2 = Image.open(path)
        updated_exif = get_exif_data(img2)
        camera, technical, _ = split_exif_sections(updated_exif)
        return jsonify({
            'success': True,
            'updated_metadata': {
                'exif': updated_exif,
                'camera': camera,
                'technical': technical,
                'gps': get_gps_info(updated_exif),
            }
        })
    except Exception as e:
        return jsonify({'error': f'Помилка редагування: {e}'}), 500


@app.route('/api/compare', methods=['GET'])
def compare():
    return jsonify({
        'metadata_extraction': [
            {
                'approach': 'PIL/Pillow (локально)',
                'speed_ms': 15, 'exif_support': True, 'iptc_support': True,
                'xmp_support': True, 'privacy': 'Повна', 'cost': 'Безкоштовно',
            },
            {
                'approach': 'ExifTool (локально)',
                'speed_ms': 45, 'exif_support': True, 'iptc_support': True,
                'xmp_support': True, 'privacy': 'Повна', 'cost': 'Безкоштовно',
            },
            {
                'approach': 'Google Cloud Vision',
                'speed_ms': 1200, 'exif_support': True, 'iptc_support': False,
                'xmp_support': False, 'privacy': 'Низька', 'cost': '$1.5/1000',
            },
            {
                'approach': 'AWS Rekognition',
                'speed_ms': 800, 'exif_support': True, 'iptc_support': False,
                'xmp_support': False, 'privacy': 'Низька', 'cost': '$0.1/img',
            },
        ],
        'ai_detection': [
            {'model': 'Hive AI', 'accuracy_percent': 92, 'speed_ms': 2000, 'false_positive_rate': 4, 'cost': '$0.005/img'},
            {'model': 'AI or Not', 'accuracy_percent': 88, 'speed_ms': 1500, 'false_positive_rate': 7, 'cost': '$0.002/img'},
            {'model': 'CNNDetection (локально)', 'accuracy_percent': 75, 'speed_ms': 500, 'false_positive_rate': 12, 'cost': 'Безкоштовно'},
            {'model': 'Метадата-евристика (цей додаток)', 'accuracy_percent': 45, 'speed_ms': 5, 'false_positive_rate': 25, 'cost': 'Безкоштовно'},
        ],
    })


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
