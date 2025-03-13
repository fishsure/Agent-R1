"""
Search tool implementation for simulating internet searches
"""

from typing import Any, Dict, List
from agent_r1.tool.tools.crag_mock_api.api import MockAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import json
from agent_r1.tool.tool_base import Tool
import os
from agent_r1.tool.tools.crag_mock_api.router import SequenceClassificationRouter
class CragApiSearchTool(Tool):
    """
    Tool for simulating internet searches using the NeuML/txtai-wikipedia model
    """
    
    def __init__(self):
        """
        Initialize the search tool
        
        Args:
            search_db: Custom search database, if None, use default
        """
        name = "api_search"
        description = "Search for information using Mock API as a knowledge source."
        parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                # "limit": {
                #     "type": "integer",
                #     "description": "Maximum number of results to return (default: 5)"
                # }
            },
            "required": ["query"]
        }
        os.environ["CRAG_MOCK_API_URL"] = "http://localhost:8000"
        self.chat_model = ChatOpenAI(model_name="gpt-4o-mini", api_key="sk-", base_url="http://localhost:8001/v1/")
        self.api = MockAPI(self.chat_model)
        self.domain_router = SequenceClassificationRouter(
            model_path="models/router/bge-m3/domain",
            classes=["finance", "music", "movie", "sports", "open"],
            device_map="auto",
        )
        
        super().__init__(name, description, parameters)

    
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
        domains = [self.domain_router(query) for query in queries]
        query_times = [x["query_time"] for x in args_list]
        kg_infos = self.api.get_kg_info(queries, query_times, domains)
        return kg_infos
        

    def _format_results(self, results: List) -> str:
        """
        Format search results for better readability
        
        Args:
            results: List of search result List
            
        Returns:
            Formatted results as a string
        """
        results_list = []
        
        for i, result in enumerate(results):
            results_list.append(self.corpus[result])
        
        return json.dumps({"results": results_list})
    
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