# --- Tool Definitions ---
# (Existing python functions search_knowledge_base/search_google remain here for reference, 
# but the schema below is what matters for the LLM)

search_knowledge_base_schema = {
  "type": "function",
  "function": {
    "name": "search_knowledge_base",
    "description": "Search internal company documents, policies and project data.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "query string"
        }
      },
      "required": ["query"]
    },
    "return": {"type": "string"}
  }
}

search_google_schema = {
  "type": "function",
  "function": {
    "name": "search_google",
    "description": "Search public information.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "query string"
        }
      },
      "required": ["query"]
    },
    "return": {"type": "string"}
  }
}

# Renamed to DEFAULT_TOOLS to imply modifiability
DEFAULT_TOOLS = [search_knowledge_base_schema, search_google_schema]
DEFAULT_SYSTEM_MSG = "You are a model that can do function calling with the following functions"
