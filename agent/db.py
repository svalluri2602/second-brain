import sqlite3

database = "bux.db"

create_table ='''CREATE TABLE IF NOT EXISTS linkedin(

id INTEGER PRIMARY KEY, 
name TEXT, 
url TEXT NOT NULL,
headline TEXT NOT NULL, 
about TEXT, 
experience TEXT, 
posts TEXT,
scraped_at TEXT
)'''


try:

    with sqlite3.connect(database) as conn:

        cursor = conn.cursor()
        cursor.execute(create_table)
        conn.commit()


except Exception as e:
    print(f"Error: {e}")
