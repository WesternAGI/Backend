#!/usr/bin/env python3

"""
AI Core Module

This module handles the core AI functionality, including:
- Communication with the OpenAI API
- Query generation with context
- Response processing
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional
import json
from dotenv import load_dotenv
from openai import OpenAI



# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from aiagent.memory.memory_manager import BaseMemoryManager, LongTermMemoryManager, ShortTermMemoryManager
from aiagent.context.reference import read_references


def query_openai(
    query: str,
    long_term_memory: LongTermMemoryManager,
    short_term_memory: ShortTermMemoryManager,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    references: Dict[str, str] = {},
    aux_data: Optional[Dict[str, Any]] = None
) -> str:
    """Send a query to OpenAI's GPT-3.5-turbo with context.

    Includes context from:
    - User profile and preferences stored in long_term_memory.json (if provided).
    - Content of files found in the 'references' directory.
    - The conversation will be summarized and stored in short_term_memory.json.

    Args:
        query: The user's question
        long_term_memory: User profile and preferences information
        references: Dictionary of reference documents to include in context
        max_tokens: Maximum tokens in response (default: 1024)
        temperature: Response randomness (0-1, default: 0.7)

    Returns:
        str: The AI's response, or an error message if the query fails.
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            return "Error: No OpenAI API key found. Set OPENAI_API_KEY in .env"

        # Build context messages from long-term memory
        messages = []

        # json to string
        user_profile_data = long_term_memory.get("user_profile")
        user_profile = json.dumps(user_profile_data if user_profile_data is not None else {})

        preferences_data = long_term_memory.get("preferences")
        preferences = json.dumps(preferences_data if preferences_data is not None else {})

        values_data = long_term_memory.get("values")
        values = json.dumps(values_data if values_data is not None else {})

        beliefs_data = long_term_memory.get("beliefs")
        beliefs = json.dumps(beliefs_data if beliefs_data is not None else {})

        aux_data_str = json.dumps(aux_data if aux_data is not None else {})
        
        past_conversations_data = short_term_memory.get("conversations")
        past_conversations = json.dumps(past_conversations_data if past_conversations_data is not None else [])
        
        active_url_data = short_term_memory.get("active_url")
        active_url = json.dumps(active_url_data if active_url_data is not None else {})

        messages.append(
            {
                "role": "system",
                "content": f"Here are the user details: \nUser Profile: {user_profile}\nPreferences: {preferences}\nValues: {values}\nBeliefs: {beliefs}\nAuxiliary Data: {aux_data_str}\nPast Conversations: {past_conversations}\nActive URL: {active_url}"
            }
        )

        print("messages:", messages) 

        # Add reference content to context
        if references:
            for filename, content in references.items():
                messages.append(
                    {
                        "role": "system",
                        "content": f"Reference - {filename}: {content[:500]}...",
                    }
                )
            
        # Add user query
        messages.append({"role": "user", "content": query})
        logging.info(f"Context messages: {messages}")
        # Query OpenAI
        logging.info("Sending query to OpenAI")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract and return the AI's response
        if response.choices and response.choices[0].message:
            ai_content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if ai_content is not None:
                logging.info("Received response content from OpenAI.")
                return ai_content
            else:
                # Content is None, log finish_reason and the message object
                logging.error(
                    f"OpenAI response error: Message content is None. Finish reason: '{finish_reason}'. Message: {response.choices[0].message.model_dump_json()}"
                )
                return f"Error querying AI: OpenAI response message content is None. Finish reason: {finish_reason}."
        else:
            # Message object itself is None or no choices
            logging.error(f"OpenAI response error: No choices or message object found. Full response: {response.model_dump_json()}")
            return "Error querying AI: No choices or message object returned from OpenAI."

    except Exception as e:
        logging.error(f"Error querying OpenAI: {e}", exc_info=True)
        return f"Error querying AI: {str(e)}"


def summarize_conversation(query: str, response: str) -> str:
    """Create a brief summary of the conversation.

    Uses the same AI model to generate a concise summary for memory storage.

    Args:
        query: The user's question
        response: The AI's answer

    Returns:
        str: A brief summary of the conversation
    """
    # Initialize client and check API key here
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            return "Summary not available - OpenAI API key not found."
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logging.exception(f"Failed to initialize OpenAI client: {e}")
        return f"Summary not available - Failed to initialize OpenAI client: {e}"

    try:
        # Create a prompt to summarize the conversation
        summary_prompt = f"""Summarize the following conversation in a single short paragraph (max 50 words):
        
        User: {query}
        AI: {response}
        
        Summary:"""

        # Make the API call with a low max_tokens to ensure brevity
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=100,
            temperature=0.5,
        )

        # Extract and clean up the summary
        summary = completion.choices[0].message.content

        # Remove any prefix the model might add like "Summary:" or "Here's a summary:"
        for prefix in ["Summary:", "Here's a summary:", "Here is a summary:"]:
            if summary.startswith(prefix):
                summary = summary[len(prefix) :].strip()

        logging.info(f"Generated conversation summary ({len(summary)} chars)")
        return summary

    except Exception as e:
        logging.error(f"Error generating conversation summary: {e}")
        return "Error generating summary."

def update_memory(query: str, response: str, memory: BaseMemoryManager):
    # Initialize client and check API key here
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.error(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            )
            return "Summary not available - OpenAI API key not found."
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logging.exception(f"Failed to initialize OpenAI client: {e}")
        return f"Summary not available - Failed to initialize OpenAI client: {e}"
    try:
        updated = False 
        # Create a prompt to summarize the conversation
        summary_prompt = f"""Given the following query, response, and memory, suggest a field to be added or updated in the memory. 
        Return only the field name and value. 
        Ex. field_name: value 
        if multiple fields are suggested, return them in a list separated by commas. 
        Ex. field_name: value, field_name: value 
        if no field is suggested, return None: None 
        follow the format exactly as shown above (without quotes or brackets (no square brackets or curly brackets) or any other characters)
        :
        User: {query}
        AI: {response}
        Memory: {memory._memory_content}
        """

        # Make the API call with a low max_tokens to ensure brevity
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=100,
            temperature=0.5,
        )

        # Extract and clean up the summary
        suggested_updates = completion.choices[0].message.content

        # parse the suggested updates
        updates = suggested_updates.split(",")

        # update the memory
        for update in updates:
            update = update.strip()
            if update.startswith("None"):
                continue
            key, value = update.split(":")
            memory.set(key.strip(), value.strip())
            updated = True

        logging.info(f"Memory updated: {memory._memory_content}")
        return updated
    except Exception as e:
        logging.exception(f"Failed to update memory: {e}")
        return False

    

def ask_ai(
    query: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    client_dir: Optional[str] = None,
    aux_data: Optional[Dict[str, Any]] = None,
    update_memory: bool = True,
) -> str:
    """
    Query the AI with a given query and optional auxiliary data.

    Args:
        query: The user's question
        aux_data: Optional auxiliary data to include in the context
        max_tokens: Maximum tokens in response
        temperature: Response randomness (0.0-1.0)
        update_memory: Whether to update memory with the conversation

    Returns:
        str: The AI's response
    """
    
    # Load long-term and short-term memory using managers
    if client_dir:
        logging.info(f"Loading memory from directory: {client_dir}")
        long_term_manager = LongTermMemoryManager(client_dir)
        short_term_manager = ShortTermMemoryManager(client_dir)
    else :
        logging.info("Loading default memory")
        long_term_manager = LongTermMemoryManager()
        short_term_manager = ShortTermMemoryManager()
    long_term_memory = long_term_manager.load()
    short_term_memory = short_term_manager.load()

    # Read all reference files
    try:
        references = read_references()
    except Exception as e:
        logging.error(f"Error loading references: {e}")
        references = {}
        
    # Check if we're running in a serverless environment
    if os.environ.get("VERCEL") and not references:
        logging.info("Running in Vercel environment, providing minimal context due to filesystem limitations")
        # Add a synthetic reference to explain the limitations
        references = {
            "system_info.txt": "This is a limited deployment running on serverless infrastructure. " +
                            "Some features requiring filesystem access may be restricted."
        }

    # Query the AI with context
    response = query_openai(
        query=query,
        long_term_memory=long_term_memory,
        short_term_memory=short_term_memory,
        aux_data=aux_data,
        references=references,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Only create summary and update memory if query was successful
    if not response.startswith("Error:") and update_memory:
        summary = summarize_conversation(query, response)
        # update short term memory: add to the conversations list
        conversations = short_term_memory.get("conversations", [])
        short_term_manager.set("conversations", conversations + [{"query": query, "response": response, "summary": summary}])
        
    else:
        logging.warning("Skipping memory update due to query error")

    return response