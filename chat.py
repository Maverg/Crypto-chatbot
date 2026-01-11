import warnings
import os
import logging
import threading
import time

from sqlalchemy import text

# warning suppressions
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")
warnings.filterwarnings("ignore", message=".*XLMRobertaSdpaSelfAttention.*")
warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed.*")

# Environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Logging configuration
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)


import json
import time  
from datetime import datetime #default is UTC
from typing import List, Dict, Any, Optional 
from getRSS import aggregate_feeds
from database import ArticleDatabase 
from vector_store import VectorStoreManager
from fetcher_service import ArticleFetcher
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import available_tools
import os
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler


#run the chatbot and type 'help' to see some of the features I have implemented


# Load environment variables
load_dotenv()

# DEBUGGING CONTROL
DEBUG_MODE = True

class SmartRSSChatbot:
    def __init__(self, days_back: int = 1, enable_fetcher: bool = True, background_fetch: bool = True):
        """Initialize chatbot with database, vector store, and tools."""
        if DEBUG_MODE:
            print("🔄 Initializing Smart RSS Chatbot...")
        
        # Initialize database first
        self.db = ArticleDatabase()
        self.days_back = days_back
        self.background_fetch_complete = False
        self.initial_fetch_done = False
        
        # Initialize vector store manager
        self.vector_manager = VectorStoreManager(
            persist_directory="./vector_store",
            max_days_to_keep=7
        )
        
        # Load existing articles from database immediately (fast)
        self._load_existing_articles()
        
        # Initialize LLM with tool calling support
        try:
            self.llm = ChatOllama(
                model="llama3.2", #Qwen/Qwen3-32B-AWQ #llama3.2 (shitty local model)
                temperature=0.1,
                num_predict=2000,  # equivalent to max_tokens
            )
            self.llm_with_tools = self.llm.bind_tools(available_tools)
        except Exception as e:
            print(f" Error initializing Ollama: {e}")
            return
        
        # Initialize memory system
        self.message_history = ChatMessageHistory()
        # Build context and chains with existing data
        
        self.context = self._build_context()
        self.agent = self._build_agent()
        
        # Start background fetching if enabled
        if enable_fetcher and background_fetch:
            self.fetcher = ArticleFetcher()
            if DEBUG_MODE:
                print("📡 Background fetcher starting...")
            self._start_background_fetch()
        elif enable_fetcher:
            # Old synchronous behavior
            self.fetcher = ArticleFetcher()
            self.fetcher.start_scheduler(interval_minutes=30)
            if DEBUG_MODE:
                print("🕐 Periodic fetcher enabled (every 30 minutes)")
            self._refresh_vector_store()
        else:
            self.fetcher = None
            if DEBUG_MODE:
                print("📖 Using cached articles only")
        
        if DEBUG_MODE:
            print("🤖 Maurice is ready with internet access!")
            print("🔍 DEBUG MODE: ON")

    def _load_existing_articles(self):
        """Quickly load existing articles from database."""
        self.display_articles = self.db.get_articles(days_back=self.days_back)
        self.vector_articles = self.display_articles
        
        if DEBUG_MODE:
            print(f"📚 Loaded {len(self.display_articles)} cached articles from database ({self.days_back} days)")
        
        # Quick vector store update with existing articles
        if self.vector_articles:
            if DEBUG_MODE:
                print("⚡ Updating vector search with cached articles...")
            stats = self.vector_manager.update_articles(self.vector_articles, silent=not DEBUG_MODE)
            total_in_store = stats['existing'] + stats['new']
            if DEBUG_MODE:
                if stats['new'] > 0:
                    print(f"✅ Vector store ready: {total_in_store} articles available")
                else:
                    print(f"✅ Vector store ready: {total_in_store} articles available")

    def _start_background_fetch(self):
        """Start background thread for fetching new articles."""
        def background_fetch_worker():
            try:
                if DEBUG_MODE:
                    print("📡 [DEBUG] Starting background fetch worker...")
                
                # Start the periodic schedule (includes initial fetch)
                self.fetcher.start_scheduler(interval_minutes=30, silent=True)

                # No manual fetch here; start_scheduler already ran one silently

                # Refresh our article cache
                old_count = len(self.display_articles)
                self.display_articles = self.db.get_articles(days_back=self.days_back)
                self.vector_articles = self.display_articles
                
                # Update vector store with any new articles
                if len(self.display_articles) > old_count:
                    stats = self.vector_manager.update_articles(self.vector_articles, silent=True)
                    if DEBUG_MODE and stats.get('new', 0) > 0:
                        print(f"\n🔍 [DEBUG] Background fetch complete: {stats['new']} new articles added")
                
                # Mark background fetch as complete
                self.background_fetch_complete = True
                self.initial_fetch_done = True
                
                # Rebuild context with fresh data
                self.context = self._build_context()
                self.agent = self._build_agent()
                
                if DEBUG_MODE:
                    print(f"\n🔍 [DEBUG] Background update complete: {len(self.display_articles)} total articles")
                
            except Exception as e:
                if DEBUG_MODE:
                    print(f"\n🔍 [DEBUG] Background fetch error: {e}")
                self.background_fetch_complete = True
        
        # Start background thread
        fetch_thread = threading.Thread(target=background_fetch_worker, daemon=True)
        fetch_thread.start()

    def get_background_status(self) -> str:
        """Get status of background fetching."""
        if not hasattr(self, 'fetcher') or not self.fetcher:
            return "📖 Background fetching disabled"
        elif not self.background_fetch_complete:
            return "📡 Background fetching in progress..."
        else:
            return "✅ Background fetching complete"

    def _build_context(self) -> str:
        """Build enhanced context with tool capabilities."""
        sources = self.get_sources()
        stats = self.get_stats()
        
        sample_articles = self.display_articles[:3]
        sample_text = "\n".join([
            f"- {article.get('title', 'No title')} (Source: {article.get('source', 'Unknown')})"
            for article in sample_articles
        ])
        
        context = f"""
You are Maurice, a helpful cryptocurrency news assistant with access to both local news articles and internet search capabilities.

AVAILABLE DATA:
- Local articles: {stats['total_articles']} from {', '.join(sources[:5])}
- Time range: {stats['time_range']}
- Real-time capabilities: Web search, current crypto prices, live news

SAMPLE RECENT ARTICLES:
{sample_text}

TOOLS AVAILABLE:
- search_web: Search the internet for current information
- get_current_crypto_price: Get real-time cryptocurrency prices
- fetch_webpage_content: Extract content from specific URLs
- search_crypto_news: Find latest crypto news from the web
- get_current_time: Get current date/time

INSTRUCTIONS:
- First check local articles for relevant information
- Use tools when you need current/real-time information
- For price queries, use get_current_crypto_price tool
- For breaking news, use search_crypto_news or search_web
- Be conversational and reference our conversation history
- Always provide accurate, up-to-date information
- When mentioning local articles, use exact URLs provided
"""
        return context

    def _build_agent(self):
        """Build the agent with conditional tool calling."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.context),
            MessagesPlaceholder("chat_history"),
            ("human", """
Current user question: {input}

Relevant local articles:
{relevant_articles}

Tool usage guidance: {tool_guidance}

{instructions}
"""),
            MessagesPlaceholder("agent_scratchpad"),
        ])
        
        # Create agent with tools
        agent = create_tool_calling_agent(self.llm, available_tools, prompt)
        
        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=available_tools,
            verbose=DEBUG_MODE,
            handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=True
        )
        
        return agent_executor

    def _build_chain(self):
        """Build the LangChain conversation chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.context),
            ("placeholder", "{chat_history}"),  # ChatMessegeHistory (ConversationSummaryBuffer)
            ("human", """
Current user question: {user_input}

Relevant articles for this query:
{relevant_articles}

Please provide a helpful response based on the available articles and our conversation history. 
When mentioning articles, use the exact article URLs provided.
""")
        ])
        
        def debug_input(x):
            if DEBUG_MODE:
                print(f"\n🔍 DEBUG - Chain Input: {x}")
            return x
        
        def debug_articles(user_input):
            articles = self._get_relevant_articles(user_input)
            if DEBUG_MODE:
                print(f"\n🔍 DEBUG - Found Articles: {len(articles.split('Article')) - 1 if 'Article' in articles else 0}")
            return articles
        
        # Create the chain
        if DEBUG_MODE:
            chain = (
                RunnableLambda(debug_input) |
                {
                    "user_input": RunnablePassthrough(),
                    "relevant_articles": RunnableLambda(debug_articles),
                    "chat_history": RunnableLambda(lambda x: self.message_history.messages)
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            chain = (
                {
                    "user_input": RunnablePassthrough(),
                    "relevant_articles": RunnableLambda(lambda x: self._get_relevant_articles(x)),
                    "chat_history": RunnableLambda(lambda x: self.message_history.messages)
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        
        return chain

    def _extract_date_from_query(self, user_input: str) -> tuple:
        """Extract date information from user query ONLY if at the beginning."""
        import re
        from dateutil.parser import parse
        from datetime import datetime
        
        user_lower = user_input.lower()
        current_year = datetime.now().year
        # Month patterns with optional day
        month_patterns = [
            r'^(january|jan)\s+(\d{1,2})',
            r'^(february|feb)\s+(\d{1,2})',
            r'^(march|mar)\s+(\d{1,2})',
            r'^(april|apr)\s+(\d{1,2})',
            r'^(may)\s+(\d{1,2})',
            r'^(june|jun)\s+(\d{1,2})',
            r'^(july|jul)\s+(\d{1,2})',
            r'^(august|aug)\s+(\d{1,2})',
            r'^(september|sept|sep)\s+(\d{1,2})',
            r'^(october|oct)\s+(\d{1,2})',
            r'^(november|nov)\s+(\d{1,2})',
            r'^(december|dec)\s+(\d{1,2})'
        ]
        
        # Check for month + day patterns
        for pattern in month_patterns:
            match = re.search(pattern, user_lower)
            if match:
                month = match.group(1)
                day = match.group(2)
                try:
                    # Parse the date with current year
                    date_str = f"{month} {day} {current_year}"
                    parsed_date = parse(date_str)
                    formatted_date = parsed_date.strftime("%Y-%m-%d")
                    
                    # Extract the remaining query (remove the date part)
                    remaining_query = re.sub(pattern, '', user_input, flags=re.IGNORECASE).strip()
                    
                    return ("exact", formatted_date, remaining_query)
                except:
                    continue

        # Check for relative dates - ONLY at start
        if re.search(r'^(?:yesterday|last\s*day)\b', user_lower):
            from datetime import timedelta
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            remaining_query = re.sub(r'^(?:yesterday|last\s*day)\b', '', user_input, flags=re.IGNORECASE).strip()
            return ("exact", yesterday, remaining_query)
        elif re.search(r'^(?:today|this\s*day)\b', user_lower):
            today = datetime.now().strftime("%Y-%m-%d")
            remaining_query = re.sub(r'^(?:today|this\s*day)\b', '', user_input, flags=re.IGNORECASE).strip()
            return ("exact", today, remaining_query)
        
        return ("range", None, user_input)

    def _get_relevant_articles(self, user_input: str) -> tuple:
        """Enhanced article search with fallback strategy. Returns (articles, should_use_tools)."""
        if DEBUG_MODE:
            print(f"\n🔍 DEBUG - Searching articles for: '{user_input}'")
        
        # Check if user is asking about news
        news_keywords = [
            'news', 'article', 'bitcoin', 'crypto', 'ethereum', 'blockchain',
            'sec', 'latest', 'recent', 'what happened', 'tell me about',
            'search', 'find', 'show me', 'any news', 'update', 'today'
        ]
        
        # Extract date information first
        date_type, date_value, remaining_query = self._extract_date_from_query(user_input)
        
        # Use remaining query for news detection
        is_news_query = any(keyword in remaining_query.lower() for keyword in news_keywords) or any(keyword in user_input.lower() for keyword in news_keywords)
        
        if DEBUG_MODE:
            print(f"🔍 DEBUG - Date detection: {date_type} = {date_value}")
            print(f"🔍 DEBUG - Remaining query: '{remaining_query}'")
            print(f"🔍 DEBUG - Is news query: {is_news_query}")
        
        if not is_news_query and date_type == "range":
            return "No specific articles needed for this general question.", False
        
        # Search local articles first
        articles = []
        if date_type == "exact" and date_value:
            # Use the remaining query for semantic search, but fall back to generic terms if empty
            search_query = remaining_query if remaining_query.strip() else "cryptocurrency bitcoin ethereum blockchain news"
            
            articles = self.vector_manager.search_articles(
                query=search_query,
                k=10,
                exact_date=date_value
            )
            if DEBUG_MODE:
                print(f"🔍 DEBUG - Exact date search found {len(articles)} articles for {date_value}")
        else:
            # Regular semantic search with range filtering
            articles = self.vector_manager.search_articles(
                user_input, 
                k=5, 
                filter_days=self.days_back
            )
            if DEBUG_MODE:
                print(f"🔍 DEBUG - Range search found {len(articles)} articles")
        
        # Determine if we should use tools as fallback
        should_use_tools = False
        
        # Use tools if:
        if len(articles) == 0:
            should_use_tools = True
            if DEBUG_MODE:
                print(f"🔍 DEBUG - No local articles found, will use web tools")
        elif any(keyword in user_input.lower() for keyword in ['current', 'price', 'now', 'today', 'latest', 'breaking']):
            # User wants current/real-time info
            should_use_tools = True
            if DEBUG_MODE:
                print(f"🔍 DEBUG - Real-time query detected, will combine local + web")
        elif date_type == "exact" and date_value == datetime.now().strftime("%Y-%m-%d"):
            # User asking about today specifically
            should_use_tools = True
            if DEBUG_MODE:
                print(f"🔍 DEBUG - Today's news requested, will use web for latest")
        
        formatted_articles = self._format_articles_for_llm(articles)
        return formatted_articles, should_use_tools

    def _format_articles_for_llm(self, articles: List[Dict]) -> str:
        """Format articles for LLM processing."""
        if not articles:
            return "No relevant articles found for the specified time range."
        
        formatted = []
        for i, article in enumerate(articles[:5], 1):
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown')
            summary = article.get('summary', 'No summary')[:300]
            published = article.get('published', 'Unknown date')
            link = article.get('link', 'No link available')
            
            formatted.append(f"""
Article {i}:
Title: {title}
Source: {source}
Published: {published}
Link: {link}
Summary: {summary}
""")
        
        return "\n".join(formatted)

    def get_sources(self) -> List[str]:
        """Get list of all sources from display articles."""
        sources = set()
        for article in self.display_articles:
            sources.add(article.get('source', 'Unknown'))
        return sorted(list(sources))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from database."""
        return self.db.get_stats(days_back=self.days_back)

    def refresh_articles(self, days_back: Optional[int] = None):
        """Refresh articles from database with new time range."""
        if days_back:
            self.days_back = days_back
        
        # Refresh vector store and articles
        self._refresh_vector_store()
        
        # Rebuild context with new data
        self.context = self._build_context()
        self.chain = self._build_chain()
        
        print(f"🔄 Refreshed: {len(self.display_articles)} articles ({self.days_back} days)")

    def chat(self, user_message: str) -> str:
        """Process user message with smart tool selection."""
        try:
            if DEBUG_MODE:
                print(f"\n🔍 DEBUG - Processing message: '{user_message}'")
                print(f"🔍 DEBUG - Background fetch status: {self.background_fetch_complete}")
            
            # Get relevant local articles and tool guidance
            relevant_articles, should_use_tools = self._get_relevant_articles(user_message)
            
            # Build dynamic instructions based on search results
            if should_use_tools:
                tool_guidance = "USE_TOOLS_RECOMMENDED"
                instructions = """
Use the available tools to get current/real-time information to supplement the local articles.
For price queries, use get_current_crypto_price.
For breaking news, use search_crypto_news or search_web.
Combine both local articles and tool results in your response.
"""
                if DEBUG_MODE:
                    print(f"🔍 DEBUG - Will use tools (reason: {'no local articles' if 'No relevant articles found' in relevant_articles else 'real-time info needed'})")
            else:
                tool_guidance = "LOCAL_ARTICLES_SUFFICIENT" 
                instructions = """
The local articles contain sufficient information to answer this question.
Focus on the local articles provided. Only use tools if absolutely necessary for current prices or breaking news.
Provide a comprehensive response based on the local articles.
"""
                if DEBUG_MODE:
                    print(f"🔍 DEBUG - Local articles sufficient, avoiding unnecessary tool calls")

            # Prepare input for agent
            agent_input = {
                "input": user_message,
                "relevant_articles": relevant_articles,
                "tool_guidance": tool_guidance,
                "instructions": instructions,
                "chat_history": self.message_history.messages
            }
            
            if DEBUG_MODE:
                print(f"🔍 DEBUG - Agent input prepared with guidance: {tool_guidance}")
            
            # True streaming via the callback manager (no sleeps)
            handler = self.TokenStreamHandler(print_tool_events=True)
            final_output = ""
            for chunk in self.agent.stream(agent_input, config={"callbacks": [handler]}):
                # TokenStreamHandler prints tokens; just capture the final output here
                if isinstance(chunk, dict) and "output" in chunk:
                    final_output = chunk["output"]

            full_response = final_output or ""
            if not full_response:
                # Fallback: get the final text if streaming didn’t surface it
                result = self.agent.invoke(agent_input)
                full_response = result.get("output", "") if isinstance(result, dict) else str(result or "")
        
            # Save to memory
            self.message_history.add_user_message(user_message)
            self.message_history.add_ai_message(full_response)
            
            return full_response
            
        except Exception as e:
            if DEBUG_MODE:
                print(f"\n🔍 DEBUG - Error occurred: {e}")
            
            error_msg = f"❌ Sorry, I encountered an error: {e}"
            print(error_msg)
            
            self.message_history.add_user_message(user_message)
            self.message_history.add_ai_message(error_msg)
            
            return error_msg

    def clear_memory(self):
        """Clear conversation memory."""
        self.message_history.clear()
        print("🧠 Conversation memory cleared!")

    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation."""
        messages = self.message_history.messages
        if not messages:
            return "No conversation history yet."
        
        summary = f"Conversation History ({len(messages)} messages):\n"
        for i, msg in enumerate(messages[-6:], 1):  # Last 6 messages
            role = "You" if isinstance(msg, HumanMessage) else "Maurice"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary += f"{i}. {role}: {content}\n"
        
        return summary

    def stream_conversation_summary(self):
        """Stream conversation summary with delays."""
        messages = self.message_history.messages
        if not messages:
            print("No conversation history yet.")
            return
        
        print(f"Conversation History ({len(messages)} messages):")
        time.sleep(0.1)
        
        for i, msg in enumerate(messages[-6:], 1):  # Last 6 messages
            role = "You" if isinstance(msg, HumanMessage) else "Maurice"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"{i}. {role}: {content}")
            time.sleep(0.2)  # Pause between messages

    def get_help(self) -> str:
        """Return enhanced help text with tool capabilities."""
        sources = ", ".join(self.get_sources())
        bg_status = self.get_background_status()
        
        help_text = f"""
🤖 **Smart News Chatbot Help**

I'm Maurice, your AI news assistant with memory, semantic search, AND internet access!

**Current Status:**
• {bg_status}
• Articles available: {len(self.display_articles)} ({self.days_back} days)

**What you can ask me:**
• "What's the latest crypto news?" (searches local + web)
• "What's the current Bitcoin price?" (real-time price)
• "Tell me about recent Ethereum developments" 
• "Search the web for SEC crypto regulations"
• "What happened with [specific topic] recently?"
• "Follow up on what we discussed earlier"

**Local Search Features:**
• 🔍 Vector-powered semantic search through {len(self.display_articles)} articles
• 🧠 Remembers our conversation
• 🔗 Provides exact article links
• 💬 Natural conversation flow

**Internet Capabilities:**
• 🌐 Real-time web search
• 💰 Current cryptocurrency prices  
• 📰 Latest breaking crypto news
• 📄 Fetch content from specific URLs
• 🕐 Current date/time information

**Special Commands:**
• "help" / "h" / "?" - Show this help message
• "stats" / "statistics" - Show database statistics
• "status" - Show background fetch status
• "[month][day] <query>" - Search specific date (e.g., "aug 4 ETH updates")
• "yesterday/today <query>" - Search recent dates
• "what did we talk about?" - Get conversation summary
• "clear memory" / "reset" - Reset conversation history

**Available Sources:** {sources}

😊😊😊
"""
        return help_text

    def stream_help(self):
        """Stream help text line by line."""
        help_lines = self.get_help().split('\n')
        for line in help_lines:
            print(line)
            time.sleep(0.05)  # Small delay for streaming effect

    def stream_stats(self):
        """Stream statistics display."""

        stats = self.get_stats()
       
        print("\n📊 **News Data Stats:**")
        time.sleep(0.1)
        print(f"• Total articles: {stats['total_articles']}")
        time.sleep(0.1)
        print(f"• Sources: {stats['sources']}")
        time.sleep(0.1)
        print(f"• Time range: {stats['time_range']}")
        time.sleep(0.1)
        print(f"• Scheduler: {'🔄 Running (every 30 min)' if self.fetcher and hasattr(self.fetcher, 'is_running') and self.fetcher.is_running else '⏹️ Stopped'}")
        time.sleep(0.1)

        
        # Calculate breakdown for top sources
        source_breakdown = stats['source_breakdown']
        total_sources = len(source_breakdown)
        top_5_sources = list(source_breakdown.items())[:5]
        
        if total_sources <= 5:
            print(f"• Source breakdown:")
        else:
            print(f"• Source breakdown (top 5 of {total_sources}):")
        time.sleep(0.1)
        
        for source, count in top_5_sources:
            print(f"  - {source}: {count} articles")
            time.sleep(0.1)
        
        # Add "etc" or summary if there are more sources
        if total_sources > 5:
            remaining_sources = total_sources - 5
            remaining_articles = sum(count for _, count in list(source_breakdown.items())[5:])
            print(f"  - ... and {remaining_sources} other sources: {remaining_articles} articles")
            time.sleep(0.1)

    def toggle_fetcher(self, enable: bool = None) -> str:
        """Enable or disable the periodic fetcher."""
        if enable is None:
            # Toggle current state
            enable = self.fetcher is None
        
        if enable and not self.fetcher:
            # Start fetcher
            self.fetcher = ArticleFetcher()
            self.fetcher.start_scheduler(interval_minutes=30)
            return "🔄 Periodic fetcher enabled (every 30 minutes)"
        elif not enable and self.fetcher:
            # Stop fetcher silently to avoid double messages
            self.fetcher.stop_scheduler(silent=True)
            self.fetcher = None
            return "⏹️ Periodic fetcher disabled"
        elif enable and self.fetcher:
            return "ℹ️ Fetcher is already running"
        else:
            return "ℹ️ Fetcher is already stopped"

    class TokenStreamHandler(BaseCallbackHandler):
        """Stream tokens and tool events to stdout in real time."""
        def __init__(self, print_tool_events: bool = True):
            self.print_tool_events = print_tool_events

        def on_llm_new_token(self, token: str, **kwargs):
            print(token, end="", flush=True)

        def on_tool_start(self, serialized, input_str, **kwargs):
            if not self.print_tool_events:
                return
            name = None
            try:
                name = (serialized or {}).get("name") or (serialized or {}).get("id")
            except Exception:
                name = None
            name = name or "tool"
            print(f"\n🔧 Using tool: {name}...", flush=True)

        def on_tool_end(self, output, **kwargs):
            if not self.print_tool_events:
                return
            print(f"\n📋 Tool result received...", flush=True)

def main():
    """Main chatbot function with cleaner startup."""
    
    days_back = 7
    
    try:
        # Initialize with background fetching enabled
        chatbot = SmartRSSChatbot(days_back=days_back, enable_fetcher=True, background_fetch=True)
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        return
    
    # Clean startup message for regular users
    print("🤖 Smart RSS News Chatbot")
    print("=" * 50)
    print("💬 Hi! I'm Maurice, your crypto news assistant!")
    print(f"📚 Ready with {len(chatbot.display_articles)} articles ({days_back} days)")
    if chatbot.fetcher:
        print("📡 Fetching latest news in background...")
    print("\nType 'help' for commands or ask about news!")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🙋 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['status', 'background status']:
                status = chatbot.get_background_status()
                print(f"📊 {status}")
                print(f"📚 Articles available: {len(chatbot.display_articles)}")
                continue
            elif user_input.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\n Thanks for chatting")
                break
            elif user_input.lower() in ['help', 'h', '?']:
                chatbot.stream_help()  
                continue
            elif user_input.lower() in ['clear memory', 'reset', 'forget']:
                chatbot.clear_memory()
                continue
            elif user_input.lower() in ['conversation', 'history', 'what did we talk about']:
                chatbot.stream_conversation_summary()
                continue
            elif user_input.lower() in ['stats', 'statistics']:
                chatbot.stream_stats()
                continue
            
            # Get AI response with streaming
            print("\n🤖 Maurice: ", end="", flush=True)
            chatbot.chat(user_input)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for chatting! 🚀")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()