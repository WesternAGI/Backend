import json
import os
import logging
from typing import Dict, List, Optional, Any
    

class BaseMemoryManager:
    """
    Abstract base class for memory management.
    
    """

    def __init__(self, memory_content: Optional[Dict] = None):
        self._memory_content = memory_content if memory_content is not None else {}


    @property
    def memory_type(self) -> str:
        raise NotImplementedError

    def set(self, key: str, value: Any) -> None:
        """Set a specific field in memory."""
        
        # safely update the memory, if the key does not exist, create it
        self._memory_content.setdefault(key, {})[key] = value
        
    def get(self, key: str) -> Optional[Any]:
        """Get a specific field from memory."""
        
        return self._memory_content.get(key, None)

class ShortTermMemoryManager(BaseMemoryManager):
    """Manages short-term memory operations.
    
    Short-term memory stores recent user interactions, current context,
    active URLs, and conversation history. This memory is more volatile
    and focuses on the immediate context of user interactions.
    
    Attributes:
        memory_file (str): Path to the short-term memory storage file
    """
    
    def __init__(self, memory_content: Optional[Dict] = None):
        super().__init__(memory_content)


    @property
    def memory_type(self) -> str:
        return "short-term"

    def update_active_url(self, url: str, title: str) -> None:
        """Update the active URL in short-term memory."""
        memory = self.load()
        memory.setdefault('active_url', {})['url'] = url
        memory['active_url']['title'] = title
        memory['active_url']['timestamp'] = self._get_timestamp()
        self.save(memory)

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat()

    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        memory = self.load()
        conversations = memory.get('conversations', [])
        return conversations[-limit:]

class LongTermMemoryManager(BaseMemoryManager):
    """Manages long-term memory operations.
    
    Long-term memory stores persistent user preferences, profile information,
    learning history, and important insights that should be remembered across
    multiple sessions. This memory provides continuity in the user experience.
    
    Attributes:
        memory_file (str): Path to the long-term memory storage file
    """
    
    def __init__(self, memory_content: Optional[Dict] = None):
        super().__init__(memory_content)


    @property
    def memory_type(self) -> str:
        return "long-term"



    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
        

if __name__ == "__main__":
    # load short-term memory
    short_term_memory = ShortTermMemoryManager()
    print(short_term_memory.load())

    # update active url
    short_term_memory.update_active_url("https://www.example.com", "Example")
    print(short_term_memory.load())

    # set a field
    short_term_memory.set("test", "value")
    print(short_term_memory.get("test"))
