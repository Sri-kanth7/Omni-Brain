import json
from typing import List, Dict, Any, Literal, TypedDict
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage

# Internal modules
import config
from retriever import SimpleInMemoryVectorStore, Document

# --- State Definitions ---
class AgentState(TypedDict):
    query: str
    original_query: str
    documents: List[Document]
    generation: str
    retry_count: int

# --- Pydantic Grader Schemas for Structured Output ---
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the query, 'yes' or 'no'"
    )

class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the retrieved documents, 'yes' or 'no'"
    )

class QueryRewriterOutput(BaseModel):
    rewritten_query: str = Field(
        description="The optimized, rewritten query for database search"
    )


class SelfRAGPipeline:
    def __init__(self, vector_store: SimpleInMemoryVectorStore):
        self.vector_store = vector_store
        self.workflow = StateGraph(AgentState)
        self._setup_graph()
        
    def _get_llm(self):
        """Helper to get LLM instance based on configuration."""
        provider = config.DEFAULT_PROVIDER
        api_key = config.get_api_key(provider)
        llm_model = config.get_llm_model(provider)
        
        if provider == "gemini":
            # Return Mock LLM if no API key is set
            if not api_key:
                return None
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key, temperature=0.0)
        elif provider == "openai":
            if not api_key:
                return None
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=llm_model, api_key=api_key, temperature=0.0)
        return None

    # --- Node 1: Retrieve ---
    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        print(f"\n--- [Node: Retrieve] Searching vector database for query: '{state['query']}' ---")
        docs = self.vector_store.similarity_search(state["query"], k=3)
        return {"documents": docs}

    # --- Node 2: Generate ---
    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        print(f"--- [Node: Generate] Generating answer based on retrieved documents ---")
        query = state["query"]
        documents = state["documents"]
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = (
            f"You are an expert analyst. Answer the user query based ONLY on the provided context below.\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {query}\n\n"
            f"If the answer cannot be found in the context, respond: 'I am sorry, but the retrieved documents do not contain the answer to your query.'\n"
            f"Answer:"
        )
        
        llm = self._get_llm()
        if llm is None:
            # Mock generator fallback
            generation = f"[MOCK ANSWER] Grounded response to: '{query}' using context: {[d.page_content[:40] + '...' for d in documents]}"
        else:
            msg = HumanMessage(content=prompt)
            res = llm.invoke([msg])
            generation = res.content
            
        return {"generation": generation}

    # --- Node 3: Rewrite Query ---
    def rewrite_query_node(self, state: AgentState) -> Dict[str, Any]:
        print(f"--- [Node: Rewrite Query] Rewriting query to improve retrieval quality ---")
        query = state["query"]
        retry_count = state.get("retry_count", 0)
        
        prompt = (
            f"Analyze the user query and output an optimized, single-line search query targeting "
            f"semantic databases. Strip away conversational words and focus on core nouns, verbs, and parameters.\n"
            f"Original Query: {query}\n"
        )
        
        llm = self._get_llm()
        if llm is None:
            rewritten_query = f"better search terms for {query}"
        else:
            try:
                structured_rewriter = llm.with_structured_output(QueryRewriterOutput)
                res = structured_rewriter.invoke([HumanMessage(content=prompt)])
                rewritten_query = res.rewritten_query
            except Exception as e:
                print(f"Query rewriter structured call failed: {e}. Falling back to text parse.")
                res = llm.invoke([HumanMessage(content=prompt)])
                rewritten_query = res.content.strip().replace('"', '')
                
        print(f"--- [Node: Rewrite Query] New Query: '{rewritten_query}' (Retry count: {retry_count + 1}) ---")
        return {"query": rewritten_query, "retry_count": retry_count + 1}

    # --- Edge Grader Nodes ---
    def grade_documents_edge(self, state: AgentState) -> str:
        """
        Grades relevance of documents.
        Returns:
            "generate" if relevant documents exist or max retries exceeded.
            "rewrite" if all documents are irrelevant and we should retry.
        """
        print(f"--- [Edge Grader] Grading document relevance ---")
        query = state["query"]
        documents = state["documents"]
        retry_count = state.get("retry_count", 0)
        
        if not documents:
            if retry_count < config.MAX_SELF_RAG_RETRIES:
                return "rewrite"
            return "generate"
            
        llm = self._get_llm()
        if llm is None:
            # Mock grading: if query contains words in the documents, mark relevant
            keywords = [w.lower() for w in query.split()]
            has_match = False
            for doc in documents:
                for kw in keywords:
                    if kw in doc.page_content.lower():
                        has_match = True
            
            score = "yes" if has_match else "no"
        else:
            try:
                structured_grader = llm.with_structured_output(GradeDocuments)
                # Combine documents text
                doc_text = "\n\n".join([d.page_content for d in documents])
                prompt = (
                    f"You are a grader. Grade if the documents are relevant to the user query.\n"
                    f"User Query: {query}\n"
                    f"Documents Context:\n{doc_text}\n"
                )
                res = structured_grader.invoke([HumanMessage(content=prompt)])
                score = res.binary_score
            except Exception as e:
                print(f"Document grader failed: {e}. Defaulting to relevant.")
                score = "yes"
                
        print(f"--- [Edge Grader] Relevance Score: {score} ---")
        if score == "yes":
            return "generate"
            
        if retry_count < config.MAX_SELF_RAG_RETRIES:
            return "rewrite"
        else:
            print("--- [Edge Grader] Maximum retries reached. Forcing generation node ---")
            return "generate"

    def grade_hallucination_edge(self, state: AgentState) -> str:
        """
        Grades answer grounding against retrieved documents.
        Returns:
            "useful" -> Answer is grounded and correct (END).
            "not grounded" -> Answer hallucinated, loop back to regenerate.
        """
        print(f"--- [Edge Grader] Grading hallucination / grounding ---")
        documents = state["documents"]
        generation = state["generation"]
        
        # If response is the default failure message, it is technically grounded/safe
        if "retrieved documents do not contain" in generation.lower() or "i am sorry" in generation.lower():
            return "useful"
            
        llm = self._get_llm()
        if llm is None:
            score = "yes"  # Mock grounded
        else:
            try:
                structured_grader = llm.with_structured_output(GradeHallucination)
                doc_text = "\n\n".join([d.page_content for d in documents])
                prompt = (
                    f"You are a grader. Grade if the LLM output is grounded / based on the retrieved context.\n"
                    f"Retrieved Context:\n{doc_text}\n\n"
                    f"LLM Output: {generation}\n"
                )
                res = structured_grader.invoke([HumanMessage(content=prompt)])
                score = res.binary_score
            except Exception as e:
                print(f"Hallucination grader failed: {e}. Defaulting to grounded.")
                score = "yes"
                
        print(f"--- [Edge Grader] Grounding Check Score: {score} ---")
        if score == "yes":
            return "useful"
        return "not grounded"

    # --- Setup LangGraph StateGraph ---
    def _setup_graph(self):
        # Add Nodes
        self.workflow.add_node("retrieve", self.retrieve_node)
        self.workflow.add_node("generate", self.generate_node)
        self.workflow.add_node("rewrite_query", self.rewrite_query_node)
        
        # Set Entrypoint
        self.workflow.set_entry_point("retrieve")
        
        # Add conditional edges
        self.workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents_edge,
            {
                "generate": "generate",
                "rewrite": "rewrite_query"
            }
        )
        
        self.workflow.add_edge("rewrite_query", "retrieve")
        
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_hallucination_edge,
            {
                "useful": END,
                "not grounded": "generate"  # Retry generating
            }
        )
        
        self.app = self.workflow.compile()

    def run(self, query: str) -> Dict[str, Any]:
        """Runs the compile LangGraph workflow."""
        inputs = {
            "query": query,
            "original_query": query,
            "documents": [],
            "generation": "",
            "retry_count": 0
        }
        return self.app.invoke(inputs)
