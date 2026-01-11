import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import hashlib

class ArticleDatabase:
    def __init__(self, db_path: str = "articles.db"):
        self.db_path = db_path
        self.init_database()
        # Auto-cleanup on initialization
        self.cleanup_old_articles()
    
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                link TEXT UNIQUE NOT NULL,
                summary TEXT,
                source TEXT NOT NULL,
                published_date DATETIME NOT NULL,
                fetched_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                content_hash TEXT UNIQUE,
                tags TEXT,  -- JSON array of tags
                category TEXT DEFAULT 'general'
            )
        ''')
        
        # Sources tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                last_fetched DATETIME,
                fetch_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Fetch logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fetch_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source_url TEXT,
                articles_found INTEGER,
                success BOOLEAN,
                error_message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_article(self, article: Dict) -> bool:
        """Add article to database, avoiding duplicates."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create content hash for deduplication
            content = f"{article['title']}{article['link']}{article['summary']}"
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR IGNORE INTO articles 
                (title, link, summary, source, published_date, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                article['title'],
                article['link'], 
                article['summary'],
                article['source'],
                article['published'],
                content_hash
            ))
            
            success = cursor.rowcount > 0
            conn.commit()
            return success
            
        except Exception as e:
            print(f"Error adding article: {e}")
            return False
        finally:
            conn.close()
    
    def get_articles(self, days_back: int = 1, source: Optional[str] = None) -> List[Dict]:
        """Retrieve articles from the last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # FIXED: Get articles from RECENT days, not old days
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        if source:
            cursor.execute('''
                SELECT title, link, summary, source, published_date 
                FROM articles 
                WHERE published_date >= ? AND source = ?
                ORDER BY published_date DESC
            ''', (cutoff_date, source))
        else:
            cursor.execute('''
                SELECT title, link, summary, source, published_date 
                FROM articles 
                WHERE published_date >= ?
                ORDER BY published_date DESC
            ''', (cutoff_date,))
        
        articles = []
        for row in cursor.fetchall():
            # Additional filtering for precise day control
            published_str = row[4]
            try:
                # Parse the stored date
                if isinstance(published_str, str):
                    from dateutil.parser import parse
                    published_date = parse(published_str)
                else:
                    published_date = published_str  # Already datetime object
                
                # Remove timezone for comparison
                if hasattr(published_date, 'replace'):
                    published_date = published_date.replace(tzinfo=None)
                
                # Only include if within the time window
                if published_date >= cutoff_date:
                    articles.append({
                        'title': row[0],
                        'link': row[1], 
                        'summary': row[2],
                        'source': row[3],
                        'published': row[4]
                    })
            except Exception as e:
                # If date parsing fails, skip this article
                print(f"Debug: Date parse error for {published_str}: {e}")
                continue
        
        conn.close()
        return articles
    
    def get_stats(self, days_back: int = 7) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Total articles
        cursor.execute('SELECT COUNT(*) FROM articles WHERE published_date >= ?', (cutoff_date,))
        total_articles = cursor.fetchone()[0]
        
        # Articles by source
        cursor.execute('''
            SELECT source, COUNT(*) 
            FROM articles 
            WHERE published_date >= ?
            GROUP BY source 
            ORDER BY COUNT(*) DESC
        ''', (cutoff_date,))
        
        source_counts = dict(cursor.fetchall())
        
        conn.close()
        return {
            'total_articles': total_articles,
            'sources': len(source_counts),
            'source_breakdown': source_counts,
            'time_range': f'Last {days_back} day{"s" if days_back != 1 else ""}'
        }
    
    def cleanup_old_articles(self, days_to_keep: int = 7) -> int:
        """Remove articles older than specified days from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Count articles to be deleted
        cursor.execute('SELECT COUNT(*) FROM articles WHERE published_date < ?', (cutoff_date,))
        count_to_delete = cursor.fetchone()[0]
        
        if count_to_delete > 0:
            # Delete old articles
            cursor.execute('DELETE FROM articles WHERE published_date < ?', (cutoff_date,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"? Database cleanup: removed {deleted_count} articles older than {days_to_keep} days")
            return deleted_count
        else:
            conn.close()
            return 0