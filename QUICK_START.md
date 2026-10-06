# Quick Start Guide - Agentic RAG with LangGraph

Get up and running with the Agentic RAG system in 5 minutes.

## Step 1: Clone or Download

```bash
# If cloning
git clone <your-repo-url>
cd agentic-rag

# Or download the files to a directory
```

## Step 2: Install Python Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

**Troubleshooting:**
- If you get version conflicts, try creating a fresh virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

## Step 3: Set Up API Keys

### Option A: Using Environment Variables (Recommended)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your keys:
   ```
   GROQ_API_KEY=your_actual_key_here
   TAVILY_API_KEY=your_actual_key_here
   PINECONE_API_KEY=your_actual_key_here
   ```

3. Save and close the file

### Option B: Manually During Notebook Execution

The notebook will prompt you to enter keys if they're not found in `.env`

**Where to get keys:**

| Service | URL | Time |
|---------|-----|------|
| Groq | https://console.groq.com | 2 min |
| Tavily | https://app.tavily.com | 2 min |
| Pinecone | https://app.pinecone.io | 3 min |

**Total setup time: ~10 minutes**

## Step 4: Launch the Notebook

```bash
jupyter notebook Agentic_RAG.ipynb
```

This opens the notebook in your browser. You're now ready to run cells!

## Step 5: Run Your First Query

Navigate to **Section 24** in the notebook and run this cell:

```python
result = ask_agent(
    "In Agentic RAG, what happens when retrieved documents are not relevant?"
)
```

You should see output like:
```
[Router] kb
[KB Retriever] Query: In Agentic RAG, what happens when retrieved documents are not relevant?
[KB Retriever] Retrieved: 4 chunks
[KB Grader] weak
[Tavily Search] Query: ...
[Tavily Search] Result characters: 7599
[Web Grader] good

==========================================================================================
QUESTION:
In Agentic RAG, what happens when retrieved documents are not relevant?

SOURCE USED:
web_search

FINAL ANSWER:
[Complete answer from web search]
==========================================================================================
```

## Common First Runs

### Test 1: Private KB Answer
```python
result = ask_agent("What is Agentic RAG?")
# Expected: Routes to KB, retrieves, grades, and answers from private knowledge base
```

### Test 2: Web Fallback
```python
result = ask_agent("What is the latest Python version?")
# Expected: KB insufficient → triggers web search → answers from Tavily
```

### Test 3: Direct Answer
```python
result = ask_agent("Hello! How are you?")
# Expected: Routes to direct → no KB/web search → simple conversational response
```

## Understanding the Output

When you run a query, you'll see:

```
[Router] kb                                          # 1. Router decision
[KB Retriever] Query: ...                            # 2. What's being searched
[KB Retriever] Retrieved: 4 chunks                   # 3. How many documents found
[KB Grader] good                                     # 4. Quality assessment
[Generate] ...                                       # 5. Answer generation
```

### Output Fields in Result:

- **SOURCE USED**: Where the answer came from (`private_kb`, `web_search`, `direct`)
- **QUESTION**: The original user question
- **FINAL ANSWER**: The generated response

## Next Steps

### 1. Explore Different Queries
Try asking various questions related to Agentic RAG, LangGraph, and web search:

```python
# More examples to try:
ask_agent("How does query rewriting help in RAG systems?")
ask_agent("What is the difference between traditional RAG and Agentic RAG?")
ask_agent("How do you evaluate evidence quality in retrieval?")
```

### 2. Modify the Knowledge Base
To use your own documents instead of LangChain docs:

1. Go to **Section 3** in the notebook
2. Change `SOURCE_URL` to your own documentation:
   ```python
   SOURCE_URL = "https://your-domain.com/docs"
   ```
3. Re-run from **Section 3** onwards

### 3. Customize the System

**Change LLM Model:**
```python
# Section 8: Initialize Groq LLM
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Try a different model
    temperature=0,
)
```

**Adjust Retrieval Settings:**
```python
# Section 6: Create Pinecone Vector Database
# Change the k value to retrieve more documents
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 8}  # Changed from 4 to 8
)
```

**Tune Web Search:**
```python
# Section 9: Initialize Tavily Web Search
web_search = TavilySearch(
    max_results=10,      # Get more results
    topic="news",        # Focus on news
    include_answer=True, # Include summaries
)
```

### 4. Deploy to Production

For production use, consider:

1. **Error Handling**: Wrap API calls in try-catch blocks
2. **Logging**: Add structured logging with timestamps
3. **Rate Limiting**: Implement request throttling
4. **Caching**: Store common queries and responses
5. **Monitoring**: Track API usage and costs
6. **Testing**: Create test cases for different query types

## Troubleshooting

### Issue: API Key Not Found
```
ValueError: Could not authenticate with GROQ_API_KEY
```
**Solution:**
- Verify `.env` file exists and has correct formatting
- Check that your API key is valid (not expired)
- Restart Jupyter after adding `.env`

### Issue: Pinecone Index Not Ready
```
PineconeException: Index not ready
```
**Solution:**
- Wait 1-2 minutes for the index to initialize
- Re-run the notebook cell
- Check your Pinecone dashboard to verify index creation

### Issue: Web Search Returns No Results
```
[Tavily Search] Result characters: 0
```
**Solution:**
- Your query might be too specific or niche
- Try rephrasing the question
- Verify your Tavily API key is valid
- Check your Tavily API usage quota

### Issue: Out of Memory
```
MemoryError or RuntimeError
```
**Solution:**
- Reduce chunk size in **Section 4**:
  ```python
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=500,  # Changed from 1000
      chunk_overlap=100,
  )
  ```
- Clear notebook kernel and restart
- Check available RAM with `free -h`

### Issue: Slow Performance
If queries are taking >15 seconds:

1. Check API status pages for Groq, Tavily, Pinecone
2. Reduce `max_results` in web search
3. Reduce `k` in retriever settings
4. Use a faster LLM model

## Performance Baselines

Expected timing for a full query:

| Path | Time | Notes |
|------|------|-------|
| Direct Answer | 0.5-1s | No retrieval needed |
| KB Answer | 2-4s | Retrieve + grade + generate |
| Web Search | 4-8s | KB insufficient + web search |

## Getting Help

### Common Issues
- Check the **Troubleshooting** section in [README.md](README.md)
- Review cell outputs for error messages
- Check API service status pages

### Learning Resources
- [LangGraph Tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [Tavily API Docs](https://docs.tavily.com/)
- [Pinecone Docs](https://docs.pinecone.io/)

### Debugging Tips
1. Add `print()` statements in node functions to see intermediate values
2. Inspect `state` dictionary to understand data flow
3. Check LLM outputs in cells before they're used
4. Verify API keys are correctly loaded:
   ```python
   import os
   print("Keys loaded:", bool(os.getenv("GROQ_API_KEY")))
   ```

## What's Next?

Once you're comfortable with the basics:

1. **Build your own KB**: Replace LangChain docs with your data
2. **Add more sources**: Connect to databases, files, APIs
3. **Customize workflows**: Add new nodes for specific use cases
4. **Deploy**: Move to production with proper error handling
5. **Monitor**: Track performance and costs

Happy building! 🚀
