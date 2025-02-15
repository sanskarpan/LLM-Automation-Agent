import aiohttp
import os
import logging
from typing import Dict, Any, List, Union, Optional
import ssl
import json
import re
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

AI_PROXY_BASE = "https://aiproxy.sanand.workers.dev/openai"

def extract_url(text: str) -> Optional[str]:
    """Extract and clean URL from text."""
    # Match URL pattern including those in quotes
    url_pattern = r'[\'"]?(https?://[^\s\'"]+)[\'"]?'
    match = re.search(url_pattern, text)
    if match:
        # Clean the URL - remove any quotes and trailing/leading whitespace
        url = match.group(1).strip().strip("'").strip('"')
        url = re.sub(r'[.,;\'"]$', '', url)
        return url
    return None

async def call_llm(prompt: str) -> str:
    """Call the LLM (GPT-4o-mini) through AI Proxy."""
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = {
                "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1  # Lower temperature for more focused responses
            }
            
            logger.info(f"Making LLM API call for task analysis")
            async with session.post(
                f"{AI_PROXY_BASE}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM API error response: {error_text}")
                    raise Exception(f"LLM API error: {error_text}")
                
                data = await response.json()
                result = data["choices"][0]["message"]["content"].strip()
                logger.info("LLM response received")
                logger.debug(f"Raw LLM response: {result}")
                return result
                
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        raise

async def analyze_language(text: str) -> Dict[str, Any]:
    """Analyze the language of the input text."""
    try:
        prompt = f"""Analyze this text and identify:
        1. Primary language used
        2. Any file paths mentioned
        3. Any specific terms that need translation
        
        Text: {text}
        
        Return JSON: {{"language": "en/hi/ta/etc", "paths": [], "terms": []}}"""
        
        response = await call_llm(prompt)
        return json.loads(response)
    except Exception as e:
        logger.error(f"Language analysis error: {e}")
        return {"language": "en", "paths": [], "terms": []}

def extract_paths(text: str) -> List[str]:
    """Extract file paths from text."""
    # Match patterns like /data/file.txt or file.txt
    path_pattern = r'(?:/data/)?[\w\-./]+\.[a-zA-Z]+'
    return re.findall(path_pattern, text)

def clean_file_path(path: str) -> str:
    """Clean file path by removing /data/ prefix and leading slashes."""
    return path.replace('/data/', '').lstrip('/')

def extract_commit_message(text: str) -> Optional[str]:
    """Extract commit message from task description."""
    # Common patterns for commit messages
    patterns = [
        r'commit message ["\'](.+?)["\']',  # commit message "..."
        r'commit ["\'](.+?)["\']',          # commit "..."
        r'message ["\'](.+?)["\']',         # message "..."
        r'with message ["\'](.+?)["\']',    # with message "..."
        r'with commit ["\'](.+?)["\']',     # with commit "..."
        r'commit message:?\s+(.+?)(?:\s+and|\s*$)',  # commit message: ... [and]
        r'with message:?\s+(.+?)(?:\s+and|\s*$)',    # with message: ... [and]
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            message = match.group(1).strip()
            # Clean up any trailing/leading quotes that might have been included
            message = message.strip('"\'')
            return message
            
    return None

async def parse_task_description(task: str, query_url: str = None) -> Dict[str, Any]:
    """Parse any task description using LLM to determine what needs to be done."""
    try:
        # Initialize parameter dictionary
        parameters = {
            'url': '',
            'query': '',
            'width': '',
            'height': '',
            'language': 'en'
        }

        # Extract email from various sources
        email = None
        if query_url:
            parsed_url = urlparse(query_url)
            query_params = parse_qs(parsed_url.query)
            email_from_query = query_params.get("user.email", [None])[0]
            if email_from_query:
                email = email_from_query

        if not email:
            # Look for ${user.email} pattern and any text after it
            user_email_pattern = r'\${user\.email}.*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            user_email_match = re.search(user_email_pattern, task)
            if user_email_match:
                email = user_email_match.group(1)
            else:
                # Try finding any email in the task
                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                email_match = re.search(email_pattern, task)
                if email_match:
                    email = email_match.group(0)

        # Clean up email if found
        if email:
            email = email.strip()
            email = re.sub(r'[.,;]$', '', email)
            parameters['email'] = email
            logger.info(f"Found email parameter: {email}")

        # Extract URL using dedicated function
        url = extract_url(task)
        if url:
            parameters['url'] = url
            logger.info(f"Found URL parameter: {url}")

        # Extract file paths
        extracted_paths = extract_paths(task)
        input_files = []
        output_files = []
        
        # Analyze potential input and output files
        for path in extracted_paths:
            clean_path = clean_file_path(path)
            if 'save' in task.lower() and clean_path == extracted_paths[-1]:
                output_files.append(clean_path)
            else:
                input_files.append(clean_path)

        # Extract commit message if present
        commit_message = extract_commit_message(task)
        if commit_message:
            parameters['commit_message'] = commit_message
            logger.info(f"Found commit message: {commit_message}")
            
        # First analyze language and extract key information
        language_info = await analyze_language(task)
        logger.info(f"Language analysis: {language_info}")
        if language_info.get('language'):
            parameters['language'] = language_info['language']
        
        # Prepare comprehensive task analysis prompt
        prompt = f"""Analyze this task and match it to one of the predefined tasks (A1-A10 or B3-B10). 
        Task is in {parameters['language']}.

        Available tasks with clear descriptions:
        A1: Install uv and run datagen.py with email parameter (Installation task)
        A2: Format markdown files using prettier@3.4.2 (Formatting task)
        A3: Count specific weekdays in dates.txt (Date counting task)
        A4: Sort contacts by last_name, first_name (Contact sorting task)
        A5: Get first lines of recent log files (Log analysis task)
        A6: Find Markdown files in docs/ and extract H1 headings to create index.json (Markdown indexing task)
        A7: Extract sender's email from email message (Email extraction task)
        A8: Extract credit card number from image (Image analysis task)
        A9: Find similar comments using embeddings (Text similarity task)
        A10: Calculate total sales for Gold tickets (Sales calculation task)

        Task keywords to task mapping:
        - If task involves "extract h1", "find headings", "create index", or "markdown index" -> A6
        - If task involves "format", "prettier", or "beautify" -> A2
        - If task involves finding similar or comparing -> A9
        - If task involves credit card or card number -> A8
        - If task involves counting days or dates -> A3

        Phase B (Business Tasks):
        B3: Fetch data from API and save it (requires URL)
        B4: Clone git repo and make commit
        B5: Run SQL query on database
        B6: Extract data from website
        B7: Compress or resize image
        B8: Transcribe audio from MP3
        B9: Convert Markdown to HTML
        B10: Create CSV filtering endpoint

        Task to analyze: "{task}"

        Return JSON with:
        {{
            "task_type": "A1-A10 or B3-B10",
            "input_files": [],
            "output_files": [],
            "parameters": {{
                "url": "",
                "query": "",
                "width": "",
                "height": ""
            }}
        }}"""
        
        # Get LLM response
        response = await call_llm(prompt)
        logger.info(f"Raw LLM response: {response}")
        
        # Extract and validate JSON
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                raise ValueError("No JSON structure found in response")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, Response: {response}")
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")

        # Merge LLM-detected parameters with our extracted parameters
        result['parameters'].update(parameters)

        # Update input/output files with our extracted paths if none were provided
        if not result.get('input_files') and input_files:
            result['input_files'] = input_files
        if not result.get('output_files') and output_files:
            result['output_files'] = output_files

        # Set default output files based on task type
        task_type = result.get('task_type', '').upper()
        if not result.get('output_files'):
            default_outputs = {
                'A3': ['dates-count.txt'],
                'A4': ['contacts-sorted.json'],
                'A5': ['logs-recent.txt'],
                'A6': ['docs/index.json'], 
                'A7': ['email-sender.txt'],
                'A8': ['credit-card.txt'],
                'A9': ['comments-similar.txt'],
                'A10': ['ticket-sales-gold.txt'],
                'B3': ['random.json'],
                'B4': ['repo'] 
            }
            result['output_files'] = default_outputs.get(task_type, [])

        # Special handling for A6
        if task_type == 'A6':
            result['input_files'] = ['docs']
            if 'docs/index.json' not in result['output_files']:
                result['output_files'] = ['docs/index.json']

        # Add commit message for B4
        if commit_message and 'B4' in task_type:
            result['parameters']['commit_message'] = commit_message


        # Clean paths
        result['input_files'] = [
            clean_file_path(path)
            for path in result.get('input_files', [])
        ]
        result['output_files'] = [
            clean_file_path(path)
            for path in result.get('output_files', [])
        ]

        # Add original task to parameters
        result['parameters']['original_task'] = task
        
        logger.info(f"Parsed task result: {result}")
        return result
            
    except Exception as e:
        logger.error(f"Error parsing task description: {str(e)}")
        raise