import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import time 

# Custom CSS Styling
st.markdown("""
<style>
    .chat-agent { font-weight: bold; color: #ffcc00; }
    .status-indicator { margin-left: 10px; font-size: 0.8em; }
    .completed { color: #00cc00; }
    .processing { color: #ffcc00; }
    .metric-box { padding: 10px; background: #2e2e2e; border-radius: 5px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Personal Financial Advisor Agent")
st.caption("🔍 Get stock market trends, company insights, and investment advice!")

# Define AI Agents with dedicated models
AGENTS = {
    "market_analyst": {
        "system_prompt": "You are MarketAnalyst. Provide key stock market trends.",
        "icon": "📈",
        "model": "llama3.2"  # Default model
    },
    "company_researcher": {
        "system_prompt": "You are CompanyResearcher. Analyze financials & risks.",
        "icon": "🏢",
        "model": "deepseek-r1:1.5b"
    },
    "investment_strategist": {
        "system_prompt": "You are InvestmentStrategist. Recommend investments.",
        "icon": "💰",
        "model": "llama3.2"
    }
}

# Initialize LLM engines for all agents
LLM_ENGINES = {
    agent_name: ChatOllama(
        model=config["model"],
        base_url="http://localhost:11434",
        temperature=0.3
    ) for agent_name, config in AGENTS.items()
}

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Advisor Configuration")
    
    # Model Selection for each agent
    st.markdown("### Agent Models")
    for agent_name, config in AGENTS.items():
        new_model = st.selectbox(
            label=f"{config['icon']} {agent_name.replace('_', ' ').title()}",
            options=["llama3.2", "deepseek-r1:1.5b"],
            index=0 if config["model"] == "llama3.2" else 1,
            key=f"model_{agent_name}"
        )
        AGENTS[agent_name]["model"] = new_model
    
    st.divider()
    
    # Performance Metrics
    st.markdown("### 🚀 Agent Performance")
    if "performance" not in st.session_state:
        st.session_state.performance = {
            agent: {"count": 0, "total_time": 0, "last_time": 0} 
            for agent in AGENTS.keys()
        }
    
    for agent, metrics in st.session_state.performance.items():
        avg_time = metrics["total_time"] / metrics["count"] if metrics["count"] > 0 else 0
        st.markdown(f"""
        <div class="metric-box">
            {AGENTS[agent]['icon']} {agent.replace('_', ' ').title()}<br>
            📊 Last: {metrics['last_time']:.2f}s<br>
            ⚡ Avg: {avg_time:.2f}s<br>
            🎯 Calls: {metrics['count']}
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🧹 Clear Metrics"):
        st.session_state.performance = {
            agent: {"count": 0, "total_time": 0, "last_time": 0} 
            for agent in AGENTS.keys()
        }
        st.rerun()

# Session State Management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hello! I’m your financial advisor. How can I help you today? 📊"}]

if "stock_plot" not in st.session_state:
    st.session_state.stock_plot = None

# Display Chat Messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Generate Response from AI Agents
def generate_agent_response(agent_name, user_query, context=""):
    start_time = time.time()
    
    # Reinitialize engine with current model selection
    llm_engine = ChatOllama(
        model=AGENTS[agent_name]["model"],
        base_url="http://localhost:11434",
        temperature=0.3
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENTS[agent_name]["system_prompt"]),
        ("human", f"User Query: {user_query}\n\nContext: {context}")
    ])
    
    chain = prompt | llm_engine | StrOutputParser()
    response = chain.invoke({})
    
    # Update performance metrics
    duration = time.time() - start_time
    st.session_state.performance[agent_name]["count"] += 1
    st.session_state.performance[agent_name]["total_time"] += duration
    st.session_state.performance[agent_name]["last_time"] = duration
    
    return response

# Fetch Stock Market Data & Generate Plots
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    
    fig, ax = plt.subplots()
    ax.plot(hist.index, hist["Close"], label="Stock Price", color='blue')
    ax.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-day MA", color='red', linestyle='dashed')
    ax.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-day MA", color='green', linestyle='dashed')
    ax.set_title(f"{ticker} Stock Price & Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (Local Currency)")
    ax.legend()
    
    st.session_state.stock_plot = fig
    
    info = stock.info
    return f"- **{info.get('longName', 'Unknown')} ({ticker})**\n" \
           f"- 📊 Price: {info.get('regularMarketPrice', 'N/A')} {info.get('currency', 'N/A')}\n" \
           f"- 📉 52W Low: {info.get('fiftyTwoWeekLow', 'N/A')} {info.get('currency', 'N/A')}\n" \
           f"- 📈 52W High: {info.get('fiftyTwoWeekHigh', 'N/A')} {info.get('currency', 'N/A')}\n" \
           f"- 💰 Market Cap: {info.get('marketCap', 'N/A')} {info.get('currency', 'N/A')}\n" \
           f"- 📅 Earnings: {info.get('earningsDate', 'N/A')}"

# Process User Query
user_query = st.chat_input("Ask your financial query here...")
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    ticker = None
    words = user_query.split()
    for word in words:
        if len(word) >= 2 and (word.isupper() or "." in word):  # Support for international tickers
            ticker = word
            break
    
    with st.spinner("📊 Analyzing market trends and investment opportunities..."):
        # Initialize execution tracking
        st.session_state.completed_agents = []
        steps = 3 if ticker else 2
        current_step = 1
        
        # Market Analyst
        st.session_state.current_agent = "MarketAnalyst"
        st.session_state.status_message = "Analyzing market trends..."
        st.session_state.progress = (current_step/steps)*100
        market_trends = generate_agent_response("market_analyst", user_query)
        st.session_state.completed_agents.append("MarketAnalyst")
        current_step += 1
        
        # Investment Strategist
        st.session_state.current_agent = "InvestmentStrategist"
        st.session_state.status_message = "Generating investment advice..."
        st.session_state.progress = (current_step/steps)*100
        investment_advice = generate_agent_response("investment_strategist", user_query, market_trends)
        st.session_state.completed_agents.append("InvestmentStrategist")
        current_step += 1
        
        # Company Researcher (if applicable)
        if ticker:
            st.session_state.current_agent = "CompanyResearcher"
            st.session_state.status_message = "Analyzing company data..."
            st.session_state.progress = (current_step/steps)*100
            stock_info = get_stock_info(ticker)
            company_analysis = generate_agent_response("company_researcher", f"Analyze {ticker}", stock_info)
            st.session_state.completed_agents.append("CompanyResearcher")
        else:
            stock_info = ""
            company_analysis = ""
        
        # Finalize processing
        st.session_state.current_agent = None
        st.session_state.status_message = "Analysis complete!"
        st.session_state.progress = 100
    
    # Compile Final Response
    final_response = f"""
- {AGENTS['market_analyst']['icon']} **Market Trends**\n  - {market_trends}\n
- {AGENTS['investment_strategist']['icon']} **Investment Advice**\n  - {investment_advice}\n
- {AGENTS['company_researcher']['icon']} **Company Analysis**\n  {stock_info}\n  {company_analysis}\n"""
    
    st.session_state.message_log.append({"role": "ai", "content": final_response})
    
    if st.session_state.stock_plot:
        st.pyplot(st.session_state.stock_plot)
    
    st.rerun()