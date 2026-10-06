# Industry-Style Agentic RAG with LangGraph

A production-ready implementation of an **Agentic Retrieval-Augmented Generation (RAG)** system that intelligently routes queries between private knowledge bases and web search, with sophisticated evidence grading and query optimization.

## Overview

This system demonstrates best practices for building intelligent RAG assistants that:
- 🎯 Check private knowledge bases **first**
- 📊 Grade evidence quality before answering
- 🌐 Fall back to web search when KB is insufficient
- 🔄 Rewrite queries for better retrieval
- 📍 Track the source of each answer
- 🛡️ Avoid hallucinations with robust grading mechanisms

## Why This Is Industry-Style

Unlike simple RAG systems that only search a single knowledge base, this implementation:

```
User Question
  ↓
Route to appropriate source (KB, web, or direct chat)
  ↓
Search private knowledge base
  ↓
Grade retrieved evidence quality
  ↓
If evidence is good → Answer from KB
  ↓
If evidence is weak → Search the web
  ↓
Grade web evidence
  ↓
If web evidence is weak → Rewrite query and retry
  ↓
Generate grounded answer with source tracking
```

## Key Features

### 🔍 Intelligent Routing
- Automatically decides whether to search KB, use web search, or answer directly
- Prevents unnecessary retrieval for simple conversational queries
- Optimizes for response speed and accuracy

### 📚 Evidence Grading
- Evaluates retrieved documents before generating answers
- Prevents confidently answering when evidence is weak
- Uses structured LLM outputs for reliable decision-making

### 🌐 Smart Web Fallback
- Only searches the web when private KB is insufficient
- Protects sensitive company information
- Seamlessly integrates live web data when needed

### 🔧 Query Rewriting
- Automatically reformulates questions for better retrieval
- Attempts to recover from weak initial searches
- Limits retries to prevent infinite loops

### 📊 Source Tracking
- Every answer includes the source: Private KB, Web Search, or Direct
- Enables transparency and trust in AI responses
- Supports auditing and compliance requirements

## Architecture

### Components

| Component | Purpose |
|-----------|---------|
| **LangChain** | Document loading, text splitting, embeddings |
| **LangGraph** | Workflow orchestration and state management |
| **HuggingFace** | Free, local embeddings (all-MiniLM-L6-v2) |
| **Pinecone** | Serverless vector database for KB retrieval |
| **Groq LLM** | Fast LLM inference (openai/gpt-oss-20b) |
| **Tavily Search** | AI-powered web search for real-time data |

### Workflow Nodes

```
route_question
    ├── retrieve_kb
    │   ├── grade_kb_evidence
    │   │   ├── generate_from_kb (if good)
    │   │   └── search_web (if weak)
    │   │       ├── grade_web_evidence
    │   │       │   ├── generate_from_web (if good)
    │   │       │   ├── rewrite_query (retry)
    │   │       │   └── answer_insufficient (if retries exhausted)
    │   │
    └── direct_answer (for greetings/simple chat)
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- API keys for: Groq, Tavily, Pinecone

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install langgraph langchain langchain-groq langchain-tavily langchain-pinecone \
            langchain-huggingface langchain-text-splitters beautifulsoup4 requests \
            sentence-transformers pinecone-client python-dotenv jupyter
```

### 2. Configure API Keys

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

The notebook will prompt for keys if not found in environment variables.

### 3. Run the Notebook

```bash
jupyter notebook Agentic_RAG.ipynb
```

## Usage

### Basic Query

```python
result = ask_agent("What is Agentic RAG and how does it work?")

# Output:
# [Router] kb
# [KB Retriever] Query: What is Agentic RAG and how does it work?
# [KB Retriever] Retrieved: 4 chunks
# [KB Grader] good
# 
# ==========================================================================================
# QUESTION:
# What is Agentic RAG and how does it work?
# 
# SOURCE USED:
# private_kb
# 
# FINAL ANSWER:
# [Detailed answer from private knowledge base]
```

### Web Fallback Example

```python
result = ask_agent("What is the latest update to Tavily Search?")

# The system will:
# 1. Search private KB (likely insufficient)
# 2. Grade evidence as weak
# 3. Trigger web search via Tavily
# 4. Grade web evidence
# 5. Generate answer from web results
# 6. Return answer with "web_search" as source
```

### Direct Answer

```python
result = ask_agent("Hello, how are you?")

# The router recognizes this as conversational
# Skips KB retrieval and web search
# Directly generates a friendly response
```

## Workflow Details

### Node 1: Route Question
**Purpose:** Determine the best source for answering  
**Decision:** KB (Agentic RAG related) vs. Direct (greetings, simple chat)

### Node 2: Retrieve KB
**Purpose:** Fetch relevant documents from vector database  
**Method:** Semantic search with 4 nearest neighbors

### Node 3: Grade KB Evidence
**Purpose:** Evaluate if KB has sufficient evidence  
**Logic:** LLM rates evidence as "good" or "weak"

### Node 4: Web Search (Conditional)
**Purpose:** Search internet if KB evidence is weak  
**Tool:** Tavily Search API with top 5 results

### Node 5: Grade Web Evidence
**Purpose:** Validate web search results  
**Logic:** LLM rates web evidence quality

### Node 6: Query Rewriting (Conditional)
**Purpose:** Reformulate question if evidence remains weak  
**Limit:** Max 1 retry to prevent infinite loops

### Nodes 7-10: Generation
**Generate from KB:** Answer using only private KB context  
**Generate from Web:** Answer using web search results  
**Direct Answer:** Simple conversational response  
**Insufficient Evidence:** Graceful failure message  

## Implementation Highlights

### Structured Outputs
```python
class RouteDecision(BaseModel):
    route: Literal["kb", "direct"]

class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"]
```
Uses Pydantic for reliable LLM output parsing via JSON mode.

### Retry Logic
```python
MAX_RETRIES = 1

if state["web_grade"] == "good":
    return "generate_from_web"
elif state["retry_count"] < MAX_RETRIES:
    return "rewrite_query"
else:
    return "answer_insufficient"
```
Prevents infinite loops while allowing one recovery attempt.

### State Management
```python
class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
```
Complete transparency into the decision-making process.

## Examples

### Example 1: KB Answer
```
Q: "What happens when retrieved documents are not relevant in Agentic RAG?"

Path: Route → Retrieve KB → Grade KB (good) → Generate from KB

Answer source: private_kb
```

### Example 2: Web Fallback
```
Q: "What is Tavily Search and why is it useful?"

Path: Route → Retrieve KB → Grade KB (weak) → Search Web → Grade Web (good) → Generate from Web

Answer source: web_search
```

### Example 3: Direct Answer
```
Q: "Hello, how are you?"

Path: Route (direct) → Generate Direct Answer

Answer source: direct
```

## Customization

### Change the Private KB
Replace the source URL in **Section 3**:
```python
SOURCE_URL = "https://your-docs-url.com/page"
```

### Adjust Retrieval Parameters
```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 8,  # Increase from 4 to get more documents
        "namespace": NAMESPACE,
    }
)
```

### Modify Grading Prompts
Edit the grader prompts in Sections 13 and 15 to match your standards.

### Switch LLMs
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

### Adjust Web Search Parameters
```python
web_search = TavilySearch(
    max_results=10,  # Increase from 5
    topic="news",    # Change from "general"
    include_answer=True,
    include_raw_content=True,  # Get full content
)
```

## Best Practices

1. **Monitor Source Distribution:** Track what percentage of answers come from KB vs. web
2. **Tune Grading Prompts:** Test various phrasings to find reliable graders
3. **Vector DB Maintenance:** Periodically update the KB with new documents
4. **Rate Limiting:** Be mindful of API quotas for Groq, Tavily, and Pinecone
5. **Error Handling:** Add try-catch blocks around external API calls
6. **Logging:** Enhance the system with structured logging for production use

## Troubleshooting

### Web search returns irrelevant results
- Adjust the query rewriting prompt to be more specific
- Increase Tavily's `max_results`
- Manually review and update your private KB

### KB grader is too lenient/strict
- Modify the grading prompt with clearer criteria
- Add examples of "good" vs. "weak" evidence
- Test with a smaller document sample first

### Infinite retries or loops
- Check `MAX_RETRIES` is set to 1
- Verify query rewriting logic produces different results
- Add explicit node transitions rather than relying on conditionals

## Performance Notes

- **Embedding:** ~2-5ms per document (HuggingFace local)
- **Vector Search:** ~100-200ms (Pinecone)
- **LLM Inference:** ~500ms-2s (Groq)
- **Web Search:** ~1-3s (Tavily)
- **Total end-to-end:** ~2-8 seconds depending on path

## Production Considerations

- Add database connection pooling
- Implement caching for repeated queries
- Add rate limiting and quota management
- Implement comprehensive logging and monitoring
- Use async/await for parallel operations
- Add user authentication and audit trails
- Implement cost tracking for API usage

## Future Enhancements

- [ ] Multi-source routing (HR docs, finance docs, etc.)
- [ ] Document metadata filtering
- [ ] Query expansion techniques
- [ ] Feedback loop for continuous improvement
- [ ] Streaming responses for large answers
- [ ] Multi-turn conversation with memory
- [ ] Citation and source highlighting
- [ ] Cost optimization and caching strategies

## References

- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/)
- [Agentic RAG Tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [Tavily Search API](https://docs.tavily.com/)
- [Pinecone Vector Database](https://docs.pinecone.io/)
- [LangChain Integration Guide](https://docs.langchain.com/oss/python/integrations/)

## License

This project is provided as-is for educational and development purposes.

## Author

Created as an industry-best-practices demonstration of Agentic RAG systems using LangGraph.
