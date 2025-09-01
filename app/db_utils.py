import sqlite3
from datetime import datetime

DB_PATH = r"C:\Users\Alawakey\Desktop\ai_customer_support\db\conversations.sqlite"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        user_message TEXT,
        bot_response TEXT,
        sentiment TEXT,
        timestamp TEXT
    )
    ''')
    conn.commit()
    conn.close()

def save_conversation(user_id, user_message, bot_response, sentiment):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
      INSERT INTO conversations(user_id, user_message, bot_response, sentiment, timestamp)
      VALUES (?,?,?,?,?)
    ''', (user_id, user_message, bot_response, sentiment, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
