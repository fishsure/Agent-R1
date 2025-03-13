"""
Search tool implementation for simulating internet searches
"""

import time
import random
from typing import Dict, List, Any, Optional

from agent_r1.tool.tool_base import Tool

from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from llama_index.core.schema import TextNode
from llama_index.retrievers.bm25 import BM25Retriever
class CragWebSearchTool(Tool):
    """
    Tool for simulating internet searches using the NeuML/txtai-wikipedia model
    """
    
    def __init__(self):
        """
        Initialize the search tool
        
        Args:
            search_db: Custom search database, if None, use default
        """
        name = "web_search"
        description = "Search for information using Web pages as a knowledge source."
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
            },
            "required": ["query"]
        }
        
        super().__init__(name, description, parameters)
        
        
        self.chunk_size = 256
        self.chunk_overlap = 0
        self.markdown_text_splitter = MarkdownTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    
    def execute(self, args: Dict) -> str:
        """
        Execute search query
        
        Args:
            args: Tool parameters, containing:
                - "query": search query string
                - "limit": optional int to limit number of results
            
        Returns:
            Formatted search results
        """
        pass
    
    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        queries = [x["query"] for x in args_list]
        search_resultss = [x["search_resultss"] for x in args_list] 
        references = []
        for query, search_results in zip(queries, search_resultss):
            chunks = [] 
            for search_result in search_results:
                text = search_result['page_result']
                snippet = search_result['page_snippet']
                if len(text) > 0:
                    chunks.extend(self.markdown_text_splitter.split_text(text))
                if len(snippet) > 0:
                    chunks.extend(self.text_splitter.split_text(snippet))
                    
            nodes = [TextNode(text=chunk) for chunk in chunks]
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
            nodes = bm25_retriever.retrieve(query)
            top_sentences = [node.get_text().strip() for node in nodes]
            if len(top_sentences) > 1:
                for snippet in top_sentences:
                    reference += "<DOC>\n"
                    reference += f"{snippet.strip()}\n"
                    reference += "</DOC>\n\n"
            elif len(top_sentences) == 1 and len(top_sentences[0]) > 0:
                reference = top_sentences[0]
            else:
                reference = "No References"
            references.append(reference)
        return references



        
    
    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for search action
        
        Args:
            args: Tool parameters
            result: Tool execution result
            
        Returns:
            Reward value
        """
        # valid tool call
        if "results" in result:
            return 0.0
        else:
            return 0.0