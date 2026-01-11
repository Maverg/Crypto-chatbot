"""
Vector store management for RSS chatbot.
Handles persistent storage, caching, and automatic cleanup of article vectors.
"""

import os
import time
from datetime import datetime, timedelta 
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from database import ArticleDatabase

DEBUG_MODE = False 

# Global cache at module level
_EMBEDDING_MODEL_CACHE = None

def get_cached_embeddings():
    """Get cached embedding model or load once."""
    global _EMBEDDING_MODEL_CACHE
    
    if _EMBEDDING_MODEL_CACHE is None:
        if DEBUG_MODE:
            print("🔄 Loading BAAI/bge-m3 embedding model...")
        start_time = time.time()
        
        _EMBEDDING_MODEL_CACHE = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        load_time = time.time() - start_time
        if DEBUG_MODE:
            print(f"✅ BAAI/bge-m3 model loaded in {load_time:.2f}s (cached)")
    else:
        if DEBUG_MODE:
            print("⚡ Using cached BAAI/bge-m3 embedding model")
    
    return _EMBEDDING_MODEL_CACHE

class VectorStoreManager:
    """Manages persistent vector store with automatic cleanup and caching."""
    
    def __init__(self, 
                 persist_directory: str = "./vector_store",
                 collection_name: str = "rss_articles_bge",  # Changed collection name for new model
                 max_days_to_keep: int = 7):
        """
        Initialize vector store manager.
        
        Args:
            persist_directory: Directory to store vectors
            collection_name: Name of the collection
            max_days_to_keep: Auto-delete articles older than this
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.max_days_to_keep = max_days_to_keep
        self.vector_store = None
        self.embeddings = None
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize components
        self._init_embeddings()
        self._load_or_create_vector_store()
    
    def _init_embeddings(self):
        """Initialize BAAI/bge-m3 embedding model using cache."""
        self.embeddings = get_cached_embeddings()
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create new one."""
        try:
            # Instantiate Chroma with persist_directory; it will reuse or create as needed
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            existing_count = self.vector_store._collection.count()
            if DEBUG_MODE:
                print(f"✅ Chroma vector store ready with BAAI/bge-m3 embeddings")
                print(f"🔍 [DEBUG] Found {existing_count} existing articles in vector store")
        except Exception as e:
            print(f"❌ Error initializing Chroma with BAAI/bge-m3: {e}")
            print("💡 Make sure BAAI/bge-m3 model is available locally")
            raise e

    def get_existing_article_ids(self) -> set:
        """Get set of article IDs already in vector store."""
        try:
            if self.vector_store._collection.count() == 0:
                return set()
            
            # Get all metadata
            results = self.vector_store._collection.get(include=['metadatas'])
            existing_ids = set()
            
            for metadata in results['metadatas']:
                if 'article_id' in metadata:
                    existing_ids.add(metadata['article_id'])
                elif 'link' in metadata:
                    # Fallback: use link as ID if article_id not available
                    existing_ids.add(metadata['link'])
            
            return existing_ids
        except Exception as e:
            if DEBUG_MODE:
                print(f"🔍 [DEBUG] Error getting existing IDs: {e}")
            return set()
    
    def update_articles(self, articles: List[Dict], force_rebuild: bool = False, silent: bool = False) -> Dict[str, int]:
        """Update vector store with optional silent mode."""
        if not articles:
            return {"existing": 0, "new": 0, "expired": 0}
        
        # Clean up old articles first
        expired_count = self._cleanup_old_articles()
        
        # CHECK FOR MISMATCH and force rebuild if needed
        vector_count = self.vector_store._collection.count()
        db_count = len(articles)
        
        if vector_count > db_count + 20:  # Allow some buffer for timing
            if DEBUG_MODE or not silent:
                print(f"Vector store mismatch detected: {vector_count} vs {db_count} database articles")
                print("🔄 Force rebuilding vector store to fix inconsistency...")
            force_rebuild = True
        
        if force_rebuild:
            if DEBUG_MODE or not silent:
                print("🔄 Force rebuilding vector store with BAAI/bge-m3...")
            self._rebuild_vector_store(articles)
            return {
                "existing": 0,
                "new": len(articles), 
                "expired": expired_count,
                "action": "rebuilt"
            }
        
        # Get existing article IDs
        existing_ids = self.get_existing_article_ids()
        
        # Find new articles
        new_articles = []
        for article in articles:
            article_id = self._get_article_id(article)
            if article_id not in existing_ids:
                new_articles.append(article)
        
        # Add new articles
        new_count = 0
        if new_articles:
            new_count = self._add_articles_to_store(new_articles, silent=silent)
            if DEBUG_MODE or not silent:
                print(f"➕ Adding {new_count} new articles...")
        
        # Statistics
        stats = {
            "existing": len(existing_ids),
            "new": new_count,
            "expired": expired_count
        }
        
        if DEBUG_MODE and not silent:
            print(f"📊 Articles analysis:")
            print(f"📊 - Existing in vector store: {stats['existing']}")
            print(f"📊 - Current in database ({self.max_days_to_keep} days): {len(articles)}")
            print(f"📊 - New to add: {stats['new']}")
            print(f"📊 - Truly expired (>{self.max_days_to_keep} days): {stats['expired']}")
        
        return stats

    def _get_article_id(self, article: Dict) -> str:
        """Generate unique ID for article."""
        # Use link as primary ID, fallback to title+source
        if article.get('link'):
            return article['link']
        else:
            title = article.get('title', 'no-title')
            source = article.get('source', 'no-source')
            return f"{source}::{title}"
    
    def _maybe_persist(self):
        """Persist to disk if the backend exposes a persist method."""
        try:
            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()
            elif hasattr(self.vector_store, "_client") and hasattr(self.vector_store._client, "persist"):
                self.vector_store._client.persist()
        except Exception as e:
            if DEBUG_MODE:
                print(f"🔍 [DEBUG] Persist skipped: {e}")

    def _add_articles_to_store(self, articles: List[Dict], silent: bool = False) -> int:
        """Add new articles to vector store with BAAI/bge-m3 embeddings."""
        try:
            documents = []
            for article in articles:
                # Create content for embedding
                content = f"{article.get('title', '')}\n{article.get('summary', '')}"
                
                # Create metadata
                metadata = {
                    "article_id": self._get_article_id(article),
                    "title": article.get('title', 'No title'),
                    "source": article.get('source', 'Unknown'),
                    "published": article.get('published', 'Unknown'),
                    "link": article.get('link', 'No link'),
                    "added_to_vector": datetime.now().isoformat(),
                    "embedding_model": "BAAI/bge-m3"
                }
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            if documents:
                # Add to vector store
                self.vector_store.add_documents(documents)
                # self.vector_store.persist()  # remove direct call
                self._maybe_persist()
                if DEBUG_MODE and not silent:
                    print(f"✅ {len(documents)} new articles embedded with BAAI/bge-m3!")
                return len(documents)
            
            return 0
            
        except Exception as e:
            if DEBUG_MODE or not silent:
                print(f"❌ Error adding articles to vector store: {e}")
            return 0

    def _cleanup_old_articles(self) -> int:
        """Remove articles older than max_days_to_keep."""
        try:
            if self.vector_store._collection.count() == 0:
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=self.max_days_to_keep)
            
            # Get all documents with metadata
            results = self.vector_store._collection.get(include=['metadatas'])
            
            ids_to_delete = []
            unparseable_dates = 0
            
            for i, metadata in enumerate(results['metadatas']):
                published_str = metadata.get('published', '')
                
                if not published_str or published_str in ['Unknown', 'None', '']:
                    # Delete articles with no valid date
                    ids_to_delete.append(results['ids'][i])
                    continue
                
                try:
                    # Use dateutil.parser for ALL dates
                    from dateutil.parser import parse
                    published_date = parse(published_str)
                    
                    # Remove timezone info for comparison
                    published_date = published_date.replace(tzinfo=None)
                    
                    if published_date < cutoff_date:
                        ids_to_delete.append(results['ids'][i])
                        
                except Exception as date_error:
                    # Count unparseable dates and delete them too
                    unparseable_dates += 1
                    ids_to_delete.append(results['ids'][i])
                    if DEBUG_MODE:
                        print(f"🔍 [DEBUG] Deleting unparseable date: {published_str}")
            
            # Delete old articles
            if ids_to_delete:
                self.vector_store._collection.delete(ids=ids_to_delete)
                # self.vector_store.persist()  # remove direct call
                self._maybe_persist()
                if DEBUG_MODE:
                    print(f"🧹 Cleaned up {len(ids_to_delete)} expired articles ({unparseable_dates} had unparseable dates)")
                return len(ids_to_delete)
            
            return 0
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"❌ Error during cleanup: {e}")
            return 0
    
    def _rebuild_vector_store(self, articles: List[Dict]):
        """Completely rebuild vector store from scratch."""
        try:
            # Delete existing collection via client if available
            try:
                if hasattr(self.vector_store, "_client"):
                    self.vector_store._client.delete_collection(name=self.collection_name)
            except Exception as _:
                pass
            
            # Recreate vector store
            self._load_or_create_vector_store()
            
            # Add all articles
            self._add_articles_to_store(articles)
            
        except Exception as e:
            print(f"❌ Error rebuilding vector store: {e}")
    
    def search_articles(self, query: str, k: int = 5, filter_days: Optional[int] = None, 
                   exact_date: Optional[str] = None) -> List[Dict]:
        """Search for relevant articles with precise date filtering."""
        try:
            if self.vector_store._collection.count() == 0:
                return []
            
            search_k = k * 3 if (filter_days or exact_date) else k
            results = self.vector_store.similarity_search(query, k=search_k)
            
            # Date filtering
            if filter_days or exact_date:
                if exact_date:
                    from dateutil.parser import parse
                    target_date = parse(exact_date).date()
                    if DEBUG_MODE:
                        print(f"🔍 [DEBUG] Filtering for exact date: {target_date}")
                else:
                    # FIXED: More precise cutoff calculation
                    cutoff_date = datetime.now() - timedelta(days=filter_days)
                    if DEBUG_MODE:
                        print(f"🔍 [DEBUG] Filtering since: {cutoff_date.strftime('%Y-%m-%d %H:%M')}")
                        print(f"🔍 [DEBUG] Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
                filtered_results = []
                for doc in results:
                    published_str = doc.metadata.get('published', '')
                    try:
                        if published_str and published_str != 'Unknown':
                            from dateutil.parser import parse
                            published_date = parse(published_str)
                            
                            if exact_date:
                                published_date_only = published_date.date()
                                if published_date_only == target_date:
                                    filtered_results.append(doc)
                                    if DEBUG_MODE:
                                        title = doc.metadata.get('title', 'No title')[:50]
                                        print(f"🔍 [DEBUG] ✅ EXACT MATCH: {published_str} - {title}")
                            else:
                                # Range filtering with precise comparison
                                published_date = published_date.replace(tzinfo=None)
                                if published_date >= cutoff_date:
                                    filtered_results.append(doc)
                                    if DEBUG_MODE:
                                        title = doc.metadata.get('title', 'No title')[:50]
                                        print(f"🔍 [DEBUG] ✅ INCLUDED: {published_str} - {title}")
                                else:
                                    if DEBUG_MODE:
                                        title = doc.metadata.get('title', 'No title')[:50]
                                        print(f"🔍 [DEBUG] ❌ EXCLUDED (too old): {published_str} - {title}")
                    except Exception as e:
                        if DEBUG_MODE:
                            print(f"🔍 [DEBUG] Date parse error: {e} for {published_str}")
                        continue
                
                results = filtered_results[:k]
                if DEBUG_MODE:
                    print(f"🔍 [DEBUG] After date filtering: {len(results)} articles")
            
            # Convert to article format
            articles = []
            for doc in results:
                articles.append({
                    "title": doc.metadata.get("title", "No title"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "published": doc.metadata.get("published", "Unknown"),
                    "link": doc.metadata.get("link", "No link"),
                    "summary": doc.page_content.split("\n")[1] if "\n" in doc.page_content else doc.page_content
                })
            
            if DEBUG_MODE:
                print(f"🔍 [DEBUG] Final search result: {len(articles)} articles")
            
            return articles
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"🔍 [DEBUG] Vector search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            count = self.vector_store._collection.count() if self.vector_store else 0
            
            # Get date range if articles exist
            date_range = "No articles"
            if count > 0:
                try:
                    results = self.vector_store._collection.get(include=['metadatas'])
                    dates = []
                    for metadata in results['metadatas']:
                        published = metadata.get('published', '')
                        if published and published != 'Unknown':
                            try:
                                # FIXED: Use dateutil.parser for ALL dates
                                from dateutil.parser import parse
                                date_obj = parse(published)
                                dates.append(date_obj)
                            except:
                                continue
                
                    if dates:
                        min_date = min(dates).strftime("%Y-%m-%d")
                        max_date = max(dates).strftime("%Y-%m-%d")
                        date_range = f"{min_date} to {max_date}"
                        
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"🔍 [DEBUG] Error getting date range: {e}")
        
            return {
                "total_vectors": count,
                "date_range": date_range,
                "max_days_kept": self.max_days_to_keep,
                "persist_directory": self.persist_directory,
                "embedding_model": "BAAI/bge-m3"  # Add model info to stats
            }
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"🔍 [DEBUG] Error getting vector stats: {e}")
            return {"total_vectors": 0, "date_range": "Error", "max_days_kept": self.max_days_to_keep}
        
    def debug_articles(self, days_filter: Optional[int] = None):
        """Debug what articles are in the vector store."""
        try:
            if self.vector_store._collection.count() == 0:
                print("🔍 [DEBUG] Vector store is empty!")
                return
            
            # Get all articles
            results = self.vector_store._collection.get(include=['metadatas'])
            print(f"🔍 [DEBUG] Total articles in vector store: {len(results['metadatas'])}")
            
            # Check dates
            from datetime import datetime, timedelta
            if days_filter:
                cutoff_date = datetime.now() - timedelta(days=days_filter)
                print(f"🔍 [DEBUG] Filtering since: {cutoff_date.strftime('%Y-%m-%d %H:%M')}")
            
            recent_count = 0
            for i, metadata in enumerate(results['metadatas'][:10]):  # Show first 10
                published = metadata.get('published', 'Unknown')
                title = metadata.get('title', 'No title')[:50]
                
                print(f"🔍 [DEBUG] Article {i+1}: {published} - {title}")
                
                if days_filter and published != 'Unknown':
                    try:
                        # FIXED: Use dateutil.parser for ALL dates
                        from dateutil.parser import parse
                        pub_date = parse(published)
                        
                        pub_date = pub_date.replace(tzinfo=None)
                        if pub_date >= cutoff_date:
                            recent_count += 1
                    except:
                        pass
                
        except Exception as e:
            print(f"🔍 [DEBUG] Error debugging vector store: {e}")