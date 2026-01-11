#must keep running (python langchain_cli/fetcher_service.py) to enable fetching in the background

import schedule
import time
import threading
from datetime import datetime
from getRSS import RSS_FEEDS, fetch_rss_feed, filter_recent_news
from database import ArticleDatabase

class ArticleFetcher:
    def __init__(self, db_path: str = "articles.db"):
        self.db = ArticleDatabase(db_path)
        self.is_running = False
        
    def fetch_all_feeds(self, silent: bool = False) -> int:
        """Fetch from all RSS feeds and store in database."""
        try:
            if not silent:
                print(f"🔄 Starting scheduled fetch at {datetime.now()}...\n🔄 Fetching...")
            
            total_new = 0
            
            for url in RSS_FEEDS:
                try:
                    entries = fetch_rss_feed(url)
                    filtered_entries = filter_recent_news(entries, url, days_back=1)
                    
                    new_articles = 0
                    for article in filtered_entries:
                        if self.db.add_article(article):
                            new_articles += 1
                    
                    total_new += new_articles
                    # print(f"📰 {url}: {new_articles} new articles")
                    
                except Exception as e:
                    if not silent:
                        print(f"❌ Error fetching {url}: {e}")
            
            if not silent:
                print(f"✅ Fetch complete: {total_new} new articles total")
            
            return total_new
        except Exception as e:
            if not silent:
                print(f"❌ Fetch failed: {e}")
            return 0

    def start_scheduler(self, interval_minutes: int = 30, silent: bool = False):  # set to every 30 minutes
        """Start the periodic fetcher."""
        if silent:
            schedule.every(interval_minutes).minutes.do(lambda: self.fetch_all_feeds(silent=True))
        else:
            schedule.every(interval_minutes).minutes.do(self.fetch_all_feeds)
        
        def run_scheduler():
            self.is_running = True
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)
        
        # Run in background thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        if not silent:
            print(f"🕐 Scheduler started: fetching every {interval_minutes} minutes")
        
        # Do initial fetch
        self.fetch_all_feeds(silent=silent)
    
    def stop_scheduler(self, silent: bool = False):
        """Stop the periodic fetcher."""
        self.is_running = False
        schedule.clear()
        if not silent:
            print("⏹️ Scheduler stopped")

def run_fetcher_service():
    """Run the fetcher as a standalone service."""
    fetcher = ArticleFetcher()
    
    try:
        fetcher.start_scheduler(interval_minutes=30)  # Fetch every 30 minutes
        print("Press Ctrl+C to stop the fetcher service...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        fetcher.stop_scheduler(silent=True)  # Use silent=True to suppress the first message
        print("👋 Fetcher service stopped")

if __name__ == "__main__":
    run_fetcher_service()