import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

DB_PATH = Path("user_models.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    """Create the users and interactions tables."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            age_group TEXT,
            gender TEXT,
            relationship TEXT,
            stage TEXT,
            diagnosis TEXT,
            diagnosis_other TEXT,
            language TEXT,
            device TEXT,
            tech_comfort TEXT,
            tech_savviness TEXT,
            has_support TEXT,
            experience INTEGER,
            occupation TEXT,
            queries_completed INTEGER DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sus_q1 INTEGER,
            sus_q2 INTEGER,
            sus_q3 INTEGER,
            sus_q4 INTEGER,
            sus_q5 INTEGER,
            sus_q6 INTEGER,
            sus_q7 INTEGER,
            sus_q8 INTEGER,
            sus_q9 INTEGER,
            sus_q10 INTEGER,
            sus_score REAL,
            sus_completed_at DATETIME
        );
    """)
    
    # Migration: Add occupation column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN occupation TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists, no migration needed
        pass
    
    # Migration: Add queries_completed column if it doesn't exist
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN queries_completed INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists, no migration needed
        pass
    
    # Migration: Add SUS columns if they don't exist
    sus_columns = [
        "sus_q1 INTEGER", "sus_q2 INTEGER", "sus_q3 INTEGER", "sus_q4 INTEGER", "sus_q5 INTEGER",
        "sus_q6 INTEGER", "sus_q7 INTEGER", "sus_q8 INTEGER", "sus_q9 INTEGER", "sus_q10 INTEGER",
        "sus_score REAL", "sus_completed_at DATETIME"
    ]
    for column_def in sus_columns:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {column_def}")
            conn.commit()
        except sqlite3.OperationalError:
            # Column already exists, no migration needed
            pass

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            question TEXT,
            answer_style TEXT,
            answer TEXT,
            answer_rating TEXT,
            source_ratings TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        );
    """)

    conn.commit()
    conn.close()

def generate_user_id():
    """Generate a unique user_id like DDMMYYYY0001."""
    date_str = datetime.now().strftime("%d%m%Y")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT MAX(CAST(SUBSTR(user_id, 9) AS INTEGER)) FROM users WHERE user_id LIKE ?",
        (f"{date_str}%",)
    )
    max_suffix = cursor.fetchone()[0]
    next_suffix = (max_suffix or 0) + 1
    user_id = f"{date_str}{str(next_suffix).zfill(4)}"
    conn.close()
    return user_id

def insert_user(
    user_id, age_group, gender, relationship, 
    stage, diagnosis, diagnosis_other, language,
    device, tech_comfort, tech_savviness, has_support, experience, occupation=None
):
    """Insert a new user and return user_id. Use provided user_id if given, otherwise generate one."""
    user_id = user_id if user_id else generate_user_id()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO users (
            user_id, 
            age_group, 
            gender,
            relationship, 
            stage,
            diagnosis,
            diagnosis_other, 
            language, 
            device,
            tech_comfort, 
            tech_savviness, 
            has_support, 
            experience,
            occupation
        ) VALUES (
            ?, 
            ?, 
            ?, 
            ?, 
            ?, 
            ?, 
            ?, 
            ?, 
            ?, 
            ?, 
            ?, 
            ?, 
            ?,
            ?)
    """, (
        user_id, 
        age_group,
        gender,
        relationship,
        stage,
        diagnosis,
        diagnosis_other,
        language, 
        device,
        tech_comfort, 
        tech_savviness, 
        has_support, 
        experience,
        occupation
    ))
    conn.commit()
    conn.close()
    return user_id

def add_interaction(user_id, question, answer_style, answer, answer_rating, source_ratings):
    """Insert a new user interaction record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO interactions (user_id, question, answer_style, answer, answer_rating, source_ratings)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, question, answer_style, answer, answer_rating, json.dumps(source_ratings))
    )
    conn.commit()
    conn.close()

def get_all_users():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df

def get_all_interactions():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM interactions", conn)
    conn.close()
    return df

def get_user(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result

def get_user_interactions(user_id):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM interactions WHERE user_id = ?", conn, params=(user_id,))
    conn.close()
    return df

def save_sus_responses(user_id, sus_responses):
    """
    Save System Usability Scale responses for a user.
    
    Args:
        user_id: The user's ID
        sus_responses: Dict with keys q1-q10, each with value 1-5
    
    Returns:
        The calculated SUS score (0-100)
    """
    # Calculate SUS score
    # For odd items (1,3,5,7,9): score = rating - 1
    # For even items (2,4,6,8,10): score = 5 - rating
    # SUS score = sum of all scores * 2.5
    
    odd_items = [1, 3, 5, 7, 9]
    score_sum = 0
    
    for i in range(1, 11):
        rating = sus_responses.get(f"q{i}", 0)
        if i in odd_items:
            score_sum += (rating - 1)
        else:
            score_sum += (5 - rating)
    
    sus_score = score_sum * 2.5
    
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE users 
        SET sus_q1 = ?, sus_q2 = ?, sus_q3 = ?, sus_q4 = ?, sus_q5 = ?,
            sus_q6 = ?, sus_q7 = ?, sus_q8 = ?, sus_q9 = ?, sus_q10 = ?,
            sus_score = ?, sus_completed_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
    """, (
        sus_responses.get("q1"), sus_responses.get("q2"), sus_responses.get("q3"),
        sus_responses.get("q4"), sus_responses.get("q5"), sus_responses.get("q6"),
        sus_responses.get("q7"), sus_responses.get("q8"), sus_responses.get("q9"),
        sus_responses.get("q10"), sus_score, user_id
    ))
    conn.commit()
    conn.close()
    
    return sus_score

def increment_user_query_count(user_id):
    """Increment the queries_completed counter for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE users 
        SET queries_completed = queries_completed + 1 
        WHERE user_id = ?
    """, (user_id,))
    conn.commit()
    conn.close()

def get_user_query_count(user_id):
    """Get the queries_completed count for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT queries_completed FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0


