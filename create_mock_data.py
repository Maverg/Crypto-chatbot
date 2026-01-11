"""
Mock data generator for testing the RSS chatbot.
Creates 3 articles for each day over the last 2 weeks.
"""

import os
import sys
import random
import sqlite3
from datetime import datetime, timedelta
from database import ArticleDatabase

# Simplified article templates with matching data keys
ARTICLE_TEMPLATES = [
    {
        "title_template": "Bitcoin Reaches New High of ${price:,} Amid {event}",
        "summary_template": "Bitcoin has surged to ${price:,} following {event}. Market analysts suggest this bullish momentum could continue as institutional adoption increases.",
        "source": "crypto.news"
    },
    {
        "title_template": "Ethereum {development} Sparks {reaction} in DeFi Markets",
        "summary_template": "The latest Ethereum {development} has caused significant {reaction} across decentralized finance platforms. Trading volumes have increased by {percentage}% over the past 24 hours.",
        "source": "cointelegraph.com"
    },
    {
        "title_template": "SEC Announces New Cryptocurrency Regulatory Framework",
        "summary_template": "The Securities and Exchange Commission has released a comprehensive regulatory framework targeting digital assets. Industry experts believe this will impact the broader crypto ecosystem significantly.",
        "source": "www.coindesk.com"
    },
    {
        "title_template": "{company} Announces ${amount}M Investment in Blockchain Technology",
        "summary_template": "{company} has revealed plans to invest ${amount} million in blockchain infrastructure over the next two years. The announcement has boosted market confidence significantly.",
        "source": "decrypt.co"
    },
    {
        "title_template": "DeFi Protocol {protocol} Suffers ${loss}M Security Breach",
        "summary_template": "Popular DeFi protocol {protocol} has experienced a ${loss} million security breach, affecting thousands of users. The team is working on security patches to resolve the situation.",
        "source": "www.financefeeds.com"
    },
    {
        "title_template": "Central Bank Digital Currency Progress in {country}",
        "summary_template": "The central bank of {country} has announced major progress regarding their digital currency initiative. This development marks a significant milestone for the global CBDC landscape.",
        "source": "blockworks.co"
    },
    {
        "title_template": "NFT Market Shows {trend} with Volume {change}",
        "summary_template": "Non-fungible token markets have demonstrated {trend} as trading volume has {change}. Popular collections like CryptoPunks have seen significant activity increases.",
        "source": "bitcoinist.com"
    },
    {
        "title_template": "Layer 2 Solution {l2_name} Achieves Record Transaction Volume",
        "summary_template": "Ethereum Layer 2 scaling solution {l2_name} has successfully processed over 1 million daily transactions. This represents a major breakthrough for blockchain scalability.",
        "source": "cryptopotato.com"
    }
]

# Sample data for template filling
SAMPLE_DATA = {
    "events": ["Institutional Adoption", "Regulatory Clarity", "Market Volatility", "Technical Breakthrough", "Economic Uncertainty"],
    "developments": ["Upgrade", "Fork", "Update", "Enhancement", "Implementation"],
    "reactions": ["Surge", "Rally", "Decline", "Volatility", "Stability"],
    "companies": ["MicroStrategy", "Tesla", "Square", "PayPal", "Visa", "Mastercard", "JPMorgan", "Goldman Sachs"],
    "protocols": ["Compound", "Aave", "Uniswap", "SushiSwap", "Curve", "Yearn", "Maker", "dYdX"],
    "countries": ["China", "USA", "Japan", "South Korea", "Singapore", "Switzerland", "Canada"],
    "trends": ["Growth", "Decline", "Recovery", "Surge", "Stability"],
    "changes": ["Up 40%", "Down 25%", "Increased", "Decreased", "Stabilized"],
    "l2_names": ["Arbitrum", "Optimism", "Polygon", "Immutable X", "Loopring", "zkSync"]
}

def clean_mock_database(db_path="mocking_articles.db"):
    """Clean all existing mock data from the database."""
    print("🧹 Cleaning existing mock data...")
    
    if not os.path.exists(db_path):
        print("  📄 No existing database found, will create new one")
        return 0
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count existing articles
        cursor.execute('SELECT COUNT(*) FROM articles')
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            # Delete all articles
            cursor.execute('DELETE FROM articles')
            conn.commit()
            print(f"  🗑️ Removed {existing_count} existing articles")
        else:
            print("  ✅ Database was already empty")
        
        conn.close()
        return existing_count
        
    except sqlite3.Error as e:
        print(f"  ❌ Error cleaning database: {e}")
        return 0

def generate_article(template, date, article_num):
    """Generate a single article from template and date."""
    
    # Generate random values
    price = random.randint(25000, 75000)
    percentage = random.randint(10, 50)
    amount = random.choice([50, 100, 250, 500, 750, 1000])
    loss = random.choice([5, 10, 15, 25, 50])
    
    # Create data dictionary with all possible keys
    data = {
        "price": price,
        "percentage": percentage,
        "amount": amount,
        "loss": loss,
        "event": random.choice(SAMPLE_DATA["events"]),
        "development": random.choice(SAMPLE_DATA["developments"]),
        "reaction": random.choice(SAMPLE_DATA["reactions"]),
        "company": random.choice(SAMPLE_DATA["companies"]),
        "protocol": random.choice(SAMPLE_DATA["protocols"]),
        "country": random.choice(SAMPLE_DATA["countries"]),
        "trend": random.choice(SAMPLE_DATA["trends"]),
        "change": random.choice(SAMPLE_DATA["changes"]),
        "l2_name": random.choice(SAMPLE_DATA["l2_names"])
    }
    
    # Fill template safely
    try:
        title = template["title_template"].format(**data)
        summary = template["summary_template"].format(**data)
    except KeyError as e:
        # Fallback if formatting fails
        title = f"Cryptocurrency News Update - {date.strftime('%Y-%m-%d')}"
        summary = f"Important cryptocurrency market developments from {date.strftime('%B %d, %Y')}. Market activity continues with various developments across the digital asset ecosystem."
    
    # Create unique link
    date_str = date.strftime("%Y-%m-%d")
    link = f"https://{template['source']}/article/{date_str}-{article_num}-{random.randint(1000, 9999)}"
    
    return {
        "title": title,
        "link": link,
        "summary": summary,
        "source": template["source"],
        "published": date.isoformat()
    }

def create_mock_database(clean_first=True):
    """Create database with mock articles for the last 2 weeks."""
    print("🔄 Creating mock database with test articles...")
    
    # Clean existing data first
    if clean_first:
        cleaned_count = clean_mock_database("mocking_articles.db")
    
    # Initialize database (will create tables if they don't exist)
    db = ArticleDatabase("mocking_articles.db")
    
    # Generate articles for last 14 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    
    total_articles = 0
    
    print("📝 Generating new mock articles...")
    for i in range(14):
        current_date = start_date + timedelta(days=i)
        
        # Create 3 articles per day
        for article_num in range(3):
            template = ARTICLE_TEMPLATES[article_num % len(ARTICLE_TEMPLATES)]
            article = generate_article(template, current_date, article_num + 1)
            
            success = db.add_article(article)
            if success:
                total_articles += 1
                print(f"📅 {current_date.strftime('%Y-%m-%d')} - Article {article_num + 1}: {article['title'][:50]}...")
    
    print(f"\n✅ Mock database created successfully!")
    print(f"📊 Total articles added: {total_articles}")
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Show statistics
    stats = db.get_stats(days_back=14)
    print(f"📰 Database stats:")
    print(f"  • Total articles: {stats['total_articles']}")
    print(f"  • Sources: {stats['sources']}")
    print(f"  • Time range: {stats['time_range']}")
    
    return db

if __name__ == "__main__":
    create_mock_database()