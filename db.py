import sqlite3
import os

def get_db():
    db_path = os.path.join(os.path.dirname(__file__), 'images.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        original_name TEXT,
        file_hash TEXT,
        file_size INTEGER,
        mime_type TEXT,
        width INTEGER,
        height INTEGER,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata_json TEXT,
        ai_score INTEGER,
        ai_analysis_json TEXT
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS metadata_edits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER,
        field_name TEXT,
        old_value TEXT,
        new_value TEXT,
        edit_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(image_id) REFERENCES images(id)
    )
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print('DB initialized')
