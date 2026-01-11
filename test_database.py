"""
Database testing and diagnostic tool for the RSS chatbot.
Use this to test database functionality and inspect stored data.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from database import ArticleDatabase

def cleanup_test_data(db, silent=False):
    """Remove test articles from database."""
    if not silent:
        print("\n🧹 Cleaning up test data...")
    
    # Delete test articles with various test identifiers
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # More comprehensive cleanup - catch various test patterns
    test_patterns = [
        "source = 'test.com'",
        "title LIKE '%DELETE ME%'",
        "title LIKE '%Test %'",
        "title LIKE '%TEST %'", 
        "summary LIKE '%[TEST ARTICLE]%'",
        "summary LIKE '%test summary%'",
        "link LIKE '%DELETE-ME%'",
        "link LIKE '%example.com%'",
        "link LIKE '%test%'",
        "title LIKE '%Bitcoin News Article%' AND source = 'test.com'"
    ]
    
    total_deleted = 0
    for pattern in test_patterns:
        cursor.execute(f'DELETE FROM articles WHERE {pattern}')
        deleted_count = cursor.rowcount
        total_deleted += deleted_count
        if not silent and deleted_count > 0:
            print(f"🗑️  Deleted {deleted_count} articles matching: {pattern}")
    
    conn.commit()
    conn.close()
    
    if not silent:
        if total_deleted > 0:
            print(f"✅ Cleanup complete! Removed {total_deleted} test articles total")
        else:
            print("ℹ️  No test data found to clean up")
    
    return total_deleted

def check_for_existing_test_data(db):
    """Check if there's existing test data in the database."""
    print("🔍 Checking for existing test data...")
    
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # Check for various test patterns
    test_checks = [
        ("Test sources", "SELECT COUNT(*) FROM articles WHERE source = 'test.com'"),
        ("DELETE ME titles", "SELECT COUNT(*) FROM articles WHERE title LIKE '%DELETE ME%'"),
        ("Example.com links", "SELECT COUNT(*) FROM articles WHERE link LIKE '%example.com%'"),
        ("Test summaries", "SELECT COUNT(*) FROM articles WHERE summary LIKE '%test summary%'")
    ]
    
    total_test_articles = 0
    for check_name, query in test_checks:
        cursor.execute(query)
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"⚠️  Found {count} articles with {check_name}")
            total_test_articles += count
    
    conn.close()
    
    if total_test_articles > 0:
        print(f"🚨 Total potential test articles found: {total_test_articles}")
        cleanup = input("🧹 Clean up existing test data now? (Y/n): ").lower()
        if cleanup != 'n':
            cleanup_test_data(db)
    else:
        print("✅ No existing test data found")

def test_basic_operations():
    """Test basic database operations with automatic cleanup."""
    print("🧪 Testing Basic Database Operations")
    print("=" * 50)
    
    db = ArticleDatabase()
    print("✅ Database initialized successfully!")
    
    # First, clean up any existing test data
    existing_cleanup = cleanup_test_data(db, silent=True)
    if existing_cleanup > 0:
        print(f"🧹 Cleaned up {existing_cleanup} existing test articles")
    
    # Create a clearly marked test article
    test_article = {
        'title': 'TEST ARTICLE - DELETE ME - Bitcoin News Test',
        'link': 'https://example.com/test-bitcoin-DELETE-ME-123',
        'summary': 'This is a test summary about Bitcoin reaching new highs. [TEST ARTICLE] - DELETE ME',
        'source': 'test.com',
        'published': datetime.now().isoformat()
    }
    
    print(f"📝 Adding test article: '{test_article['title'][:50]}...'")
    success = db.add_article(test_article)
    print(f"✅ Article added: {success}")
    
    # Test duplicate prevention
    duplicate_success = db.add_article(test_article)
    print(f"✅ Duplicate prevention: {not duplicate_success} (should be True)")
    
    # Verify test article exists
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM articles WHERE source = 'test.com'")
    test_count = cursor.fetchone()[0]
    conn.close()
    print(f"✅ Test articles in database: {test_count}")
    
    return db, test_article

def test_queries(db):
    """Test various database queries."""
    print("\n🔍 Testing Database Queries")
    print("=" * 50)
    
    # Test getting articles
    for days in [1, 3, 7]:
        articles = db.get_articles(days_back=days)
        print(f"📰 Articles (last {days} day{'s' if days > 1 else ''}): {len(articles)}")
    
    # Test stats
    stats = db.get_stats()
    print(f"\n📊 Database Stats:")
    print(f"  • Total articles: {stats['total_articles']}")
    print(f"  • Sources: {stats['sources']}")
    print(f"  • Time range: {stats['time_range']}")
    if stats['source_breakdown']:
        top_source = max(stats['source_breakdown'].items(), key=lambda x: x[1])
        print(f"  • Top source: {top_source[0]} ({top_source[1]} articles)")

def inspect_database(db):
    """Inspect database contents."""
    print("\n🔍 Database Inspection")
    print("=" * 50)
    
    articles = db.get_articles(days_back=7)
    
    if not articles:
        print("❌ No articles found in database!")
        return
    
    print(f"📰 Found {len(articles)} articles in last 7 days")
    print("\n📋 Sample Articles:")
    
    for i, article in enumerate(articles[:5], 1):
        title = article['title']
        is_test = any(marker in title.upper() for marker in ['TEST', 'DELETE ME']) or article['source'] == 'test.com'
        marker = " [TEST]" if is_test else ""
        
        print(f"\n{i}. {title[:60]}...{marker}")
        print(f"   Source: {article['source']}")
        print(f"   Published: {article['published']}")
        print(f"   Link: {article['link'][:50]}...")

def view_all_database(db):
    """View entire database contents with pagination."""
    print("\n📚 Complete Database View")
    print("=" * 50)
    
    # Get all articles from database directly (no time limit)
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT title, source, published_date, link, summary 
        FROM articles 
        ORDER BY published_date DESC
    ''')
    
    all_articles = cursor.fetchall()
    conn.close()
    
    if not all_articles:
        print("❌ No articles found in database!")
        return
    
    # Count test articles
    test_count = sum(1 for article in all_articles 
                    if 'test.com' in str(article).lower() or 
                       'delete me' in str(article).lower() or
                       'example.com' in str(article).lower())
    
    print(f"📰 Total articles in database: {len(all_articles)}")
    if test_count > 0:
        print(f"⚠️  Test articles detected: {test_count}")
    
    # Pagination
    page_size = 10
    page = 0
    
    while True:
        start = page * page_size
        end = start + page_size
        page_articles = all_articles[start:end]
        
        if not page_articles:
            print("\n📝 No more articles to display.")
            break
        
        print(f"\n📄 Page {page + 1} ({start + 1}-{min(end, len(all_articles))} of {len(all_articles)})")
        print("-" * 60)
        
        for i, article in enumerate(page_articles, start + 1):
            title, source, published, link, summary = article
            
            # Mark test articles
            is_test = (source == 'test.com' or 
                      'delete me' in title.lower() or 
                      'example.com' in link.lower() or
                      'test' in title.lower())
            marker = " [TEST]" if is_test else ""
            
            print(f"\n{i}. {title}{marker}")
            print(f"   📰 Source: {source}")
            print(f"   📅 Published: {published}")
            print(f"   🔗 Link: {link}")
            print(f"   📝 Summary: {summary[:100]}...")
        
        # Ask user what to do next
        choice = input(f"\n[n]ext, [p]revious, [c]leanup test data, [q]uit: ").lower().strip()
        
        if choice == 'n':
            page += 1
        elif choice == 'p' and page > 0:
            page -= 1
        elif choice == 'c':
            deleted = cleanup_test_data(db)
            if deleted > 0:
                print("🔄 Refreshing view...")
                return view_all_database(db)  # Restart with clean data
        elif choice == 'q':
            break
        else:
            print("Invalid choice! Use n/p/c/q")

def test_performance(db):
    """Test database performance with multiple operations."""
    print("\n⚡ Performance Testing")
    print("=" * 50)
    
    import time
    
    # Test query performance
    start_time = time.time()
    articles = db.get_articles(days_back=7)
    query_time = time.time() - start_time
    print(f"⏱️  Query time (7 days): {query_time:.3f}s for {len(articles)} articles")
    
    # Test stats performance
    start_time = time.time()
    stats = db.get_stats(days_back=7)
    stats_time = time.time() - start_time
    print(f"⏱️  Stats time: {stats_time:.3f}s")
    
    # Test large query performance
    start_time = time.time()
    all_articles = db.get_articles(days_back=30)  # Try 30 days
    large_query_time = time.time() - start_time
    print(f"⏱️  Large query time (30 days): {large_query_time:.3f}s for {len(all_articles)} articles")

def check_database_file():
    """Check database file status."""
    print("🗄️  Database File Information")
    print("=" * 50)
    
    db_path = "articles.db"
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        print(f"📁 Database file: {db_path}")
        print(f"💾 File size: {size:,} bytes ({size/1024:.1f} KB)")
        
        # Get modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(db_path))
        print(f"🕐 Last modified: {mod_time}")
        
        # Get table info
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count total articles
        cursor.execute('SELECT COUNT(*) FROM articles')
        total_articles = cursor.fetchone()[0]
        print(f"📰 Total articles: {total_articles}")
        
        # Count test articles
        cursor.execute("SELECT COUNT(*) FROM articles WHERE source = 'test.com' OR title LIKE '%DELETE ME%' OR link LIKE '%example.com%'")
        test_articles = cursor.fetchone()[0]
        if test_articles > 0:
            print(f"⚠️  Test articles: {test_articles}")
        
        # Get date range
        cursor.execute('SELECT MIN(published_date), MAX(published_date) FROM articles')
        date_range = cursor.fetchone()
        if date_range[0] and date_range[1]:
            print(f"📅 Date range: {date_range[0]} to {date_range[1]}")
        
        conn.close()
    else:
        print(f"❌ Database file not found: {db_path}")

def database_maintenance():
    """Perform database maintenance tasks."""
    print("\n🔧 Database Maintenance")
    print("=" * 50)
    
    db = ArticleDatabase()
    
    # First, clean up test data
    cleanup_test_data(db)
    
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    # Find and show potential duplicates
    cursor.execute('''
        SELECT title, COUNT(*) as count 
        FROM articles 
        GROUP BY title 
        HAVING COUNT(*) > 1
        ORDER BY count DESC
    ''')
    
    duplicates = cursor.fetchall()
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} potential duplicate titles:")
        for title, count in duplicates[:5]:
            print(f"   • '{title[:50]}...' appears {count} times")
    else:
        print("\n✅ No duplicate titles found")
    
    # Show sources with article counts
    cursor.execute('''
        SELECT source, COUNT(*) as count 
        FROM articles 
        GROUP BY source 
        ORDER BY count DESC
    ''')
    
    sources = cursor.fetchall()
    print(f"\n📊 Articles by source (top 10):")
    for source, count in sources[:10]:
        marker = " [TEST]" if source == 'test.com' else ""
        print(f"   • {source}{marker}: {count} articles")
    
    conn.close()

def main():
    """Run all tests with interactive menu."""
    print("🧪 RSS Chatbot Database Test Suite")
    print("=" * 60)
    
    # Always check for existing test data first
    db = ArticleDatabase()
    check_for_existing_test_data(db)
    
    while True:
        print("\n🔧 Choose a test option:")
        print("1. Run basic tests (auto-cleanup)")
        print("2. View database contents (paginated)")
        print("3. Check database file info")
        print("4. Performance testing")
        print("5. Database maintenance + cleanup")
        print("6. Test fetcher service")
        print("7. Manual cleanup test data")
        print("8. Run all tests (with cleanup)")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        try:
            if choice == '1':
                db, test_article = test_basic_operations()
                test_queries(db)
                inspect_database(db)
                # Auto cleanup after tests
                print("\n🧹 Auto-cleaning test data...")
                cleanup_test_data(db)
                
            elif choice == '2':
                db = ArticleDatabase()
                view_all_database(db)
                
            elif choice == '3':
                check_database_file()
                
            elif choice == '4':
                db = ArticleDatabase()
                test_performance(db)
                
            elif choice == '5':
                database_maintenance()
                
            elif choice == '6':
                test_fetcher()
                
            elif choice == '7':
                db = ArticleDatabase()
                cleanup_test_data(db)
                
            elif choice == '8':
                # Run all tests with cleanup
                check_database_file()
                db, test_article = test_basic_operations()
                test_queries(db)
                inspect_database(db)
                test_performance(db)
                database_maintenance()
                
                if input("\n🔄 Test the fetcher service? (y/N): ").lower().startswith('y'):
                    test_fetcher()
                
                # Always cleanup after full test suite
                print("\n🧹 Final cleanup...")
                cleanup_test_data(db)
                print("\n✅ All tests completed successfully!")
                
            elif choice == '9':
                # Cleanup before exit
                db = ArticleDatabase()
                cleanup_count = cleanup_test_data(db, silent=True)
                if cleanup_count > 0:
                    print(f"🧹 Cleaned up {cleanup_count} test articles before exit")
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice! Please enter 1-9.")
                
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

def test_fetcher():
    """Test the article fetcher."""
    print("\n🔄 Testing Article Fetcher")
    print("=" * 50)
    
    try:
        from Mocking.fetcher_service import ArticleFetcher
        
        fetcher = ArticleFetcher()
        print("✅ Fetcher initialized")
        
        # Show before stats
        db = ArticleDatabase()
        before_stats = db.get_stats()
        print(f"📊 Articles before fetch: {before_stats['total_articles']}")
        
        # Run one fetch cycle
        new_articles = fetcher.fetch_all_feeds()
        print(f"📰 Fetched {new_articles} new articles")
        
        # Show after stats
        after_stats = db.get_stats()
        print(f"📊 Articles after fetch: {after_stats['total_articles']}")
        print(f"➕ Net new articles: {after_stats['total_articles'] - before_stats['total_articles']}")
        
    except ImportError as e:
        print(f"❌ Could not import fetcher: {e}")
    except Exception as e:
        print(f"❌ Fetcher test failed: {e}")

if __name__ == "__main__":
    main()