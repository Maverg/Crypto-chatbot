import feedparser
from datetime import datetime, timedelta, timezone  # use stdlib timezone
from dateutil.parser import parse
import requests
from urllib.parse import urlparse


# List of RSS feed URLs from the original n8n workflow
#Feeds commented out were not working
#Feel free to add more feeds to this list but test if they work first
#to test the feeds just uncomment the debugging in main() run it and look at the status codes (200) and entries
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss",
    "https://decrypt.co/feed",
    "https://cryptobriefing.com/feed/",
    "https://crypto.news/feed/",
    "https://bitcoinist.com/feed/",
    "https://cryptopotato.com/feed/",
    "https://www.sec.gov/news/pressreleases.rss",
    "https://www.coincenter.org/feed",
    "http://feeds.finra.org/FINRANotices",
    "http://feeds.finra.org/FINRANews",
    "https://www.gov.uk/government/organisations/financial-conduct-authority.atom",
    "https://www.bleepingcomputer.com/feed/",
    #"https://www.theblock.co/feed", # Returns 403
    #"https://www.financefeeds.com/feed/",
    "https://www.fca.org.uk/news/rss",
    "https://www.gov.uk/government/organisations/hm-treasury.atom",
    # "https://www.databreachtoday.com/rss-feeds",  # Returns 429
    # "https://thehackernews.com/feed",  # Returns 403
    "https://blockworks.co/feed",  # Handled encoding
    # "https://www.bitcoinmarketjournal.com/feed/",  # Malformed XML
    "https://coinjournal.net/feed",
    #"https://cryptoslate.com/feed/", # Returns 403
]

failed = []

def fetch_rss_feed(url):
    """Fetch and parse an RSS feed, checking for valid HTTP response."""
    feed_name = urlparse(url).netloc or url
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, allow_redirects=True, headers=headers)
        status_code = response.status_code
        
        if status_code != 200:
            print(f"Skipping {feed_name}: Non-200 status code {status_code}")
            failed.append(feed_name)
            return []
        
        feed = feedparser.parse(response.content)
        
        if feed.bozo and 'encoding' not in str(feed.bozo_exception).lower():
            print(f"Error parsing feed {feed_name}: {feed.bozo_exception}") 
            return []
        elif feed.bozo:
            print(f"Warning: Feed {feed_name} has encoding issues but continuing: {feed.bozo_exception}") 
            pass
        
        return feed.entries
        
    except requests.RequestException as e:
        print(f"Error fetching feed {feed_name}: {e}")
        failed.append(feed_name)
        return []
    except Exception as e:
        print(f"Unexpected error with feed {feed_name}: {e}")  
        failed.append(feed_name)
        return []

def filter_recent_news(entries, source_url, days_back):
    """Filter feed entries to only those from the last X days."""
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back) 
    filtered_entries = []
    feed_name = urlparse(source_url).netloc or source_url
    
    for entry in entries:
        try:
            pub_date_str = (entry.get('published', '') or 
                            entry.get('pubDate', '') or 
                            entry.get('updated', '') or
                            entry.get('created', ''))
            
            if not pub_date_str:
                pub_date = datetime.now(timezone.utc)
                pub_date_str = pub_date.isoformat()
            else:
                dt = parse(pub_date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                pub_date = dt.astimezone(timezone.utc)
                pub_date_str = pub_date.isoformat()

            if pub_date >= cutoff_date:
                summary = entry.get('summary', '') or entry.get('description', '')
                if len(summary) > 500:
                    summary = summary[:500] + "..."
                
                filtered_entries.append({
                    'title': entry.get('title', 'No title available'),
                    'link': entry.get('link', ''),
                    'published': pub_date_str,
                    'summary': summary,
                    'source': feed_name
                })
        except Exception as e:
            print(f"Error processing entry in {feed_name} titled '{entry.get('title', 'unknown')}': {e}")
            continue
    
    return filtered_entries


def aggregate_feeds(days_back):
    """Main function to fetch, merge, filter, and aggregate RSS feeds."""
    print("🔍 Looking for new articles...")
    all_entries = []
    
    # Fetch all feeds
    for url in RSS_FEEDS:
        entries = fetch_rss_feed(url)
        filtered_entries = filter_recent_news(entries, url, days_back)
        all_entries.extend(filtered_entries)
    
    print(f"✅ Fetch complete: {len(all_entries)} new articles")
    
    # Aggregate all data
    aggregated_data = {
        "total_items": len(all_entries),
        "items": all_entries
    }
    
    return aggregated_data

def main():
    """"""
    #Run the RSS aggregator and print results.

    print("Starting RSS feed aggregation...") 
    result = aggregate_feeds(days_back=1)
    print(f"\nTotal items found: {result['total_items']}")  # Now handled in aggregate_feeds
    
    #Show failed feeds only if there are any (optional)
    if failed:
        print(f"Failed feeds: {', '.join(failed)}")
    
    if result['total_items'] == 0:
        print("No items found. Check logs for errors with specific feeds.")


    if result['total_items'] > 0:
        for item in result['items']:
            print(f"\nTitle: {item['title']}")
            print(f"Link: {item['link']}")
            print(f"Published: {item['published']}")
            print(f"Summary: {item['summary'][:200]}...")  # Truncate summary
            print(f"Source: {item['source']}")
    else:
        print("No items found. Check logs for errors with specific feeds.")
""""""

#testing feeds
"""
feeds_to_test = [
    "https://www.gov.uk/government/organisations/financial-conduct-authority.atom",
    "https://www.bleepingcomputer.com/feed/",
    "https://www.theblock.co/feed",
]

for url in feeds_to_test:
    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            print(f"⚠️ Invalid feed: {url} - {feed.bozo_exception}")
        else:
            print(f"✅ Valid feed: {url} - {len(feed.entries)} entries")
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
"""

if __name__ == "__main__":
    main()