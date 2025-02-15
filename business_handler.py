import aiohttp
import json
import sqlite3
from bs4 import BeautifulSoup
from PIL import Image
import markdown2
import csv
import base64
from fastapi import FastAPI, HTTPException
from pathlib import Path
from typing import List, Dict, Any
import logging
import git
from llm_helper import call_llm

logger = logging.getLogger(__name__)

class BusinessTaskHandler:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def _validate_path(self, path: str) -> Path:
        """Basic path validation"""
        return Path(path).resolve()

    async def handle_b3(self, input_files: List[str], output_files: List[str], **kwargs):
        """Fetch data from API and save it"""
        try:
            url = kwargs.get('url')
            if not url:
                raise ValueError("URL parameter is required")

            output_path = self._validate_path(output_files[0])

            # Fetch data
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"API request failed: {response.status}")
                    data = await response.json()

            # Save data
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error in B3: {str(e)}", exc_info=True)
            return False

    async def handle_b4(self, input_files: List[str], output_files: List[str], **kwargs):
        """Clone git repo and make commit"""
        try:
            repo_url = kwargs.get('repo_url')
            commit_message = kwargs.get('commit_message', 'Update files')
            target_dir = self._validate_path(output_files[0])

            # Clone repo
            repo = git.Repo.clone_from(repo_url, target_dir)

            # Make changes if specified
            if 'file_changes' in kwargs:
                for file_path, content in kwargs['file_changes'].items():
                    full_path = target_dir / file_path
                    with open(full_path, 'w') as f:
                        f.write(content)

            # Commit changes
            repo.index.add('*')
            repo.index.commit(commit_message)

            return True
        except Exception as e:
            logger.error(f"Error in B4: {str(e)}", exc_info=True)
            return False

    async def handle_b5(self, input_files: List[str], output_files: List[str], **kwargs):
        """Run SQL query on SQLite/DuckDB database"""
        try:
            query = kwargs.get('query')
            if not query:
                raise ValueError("SQL query is required")

            db_path = self._validate_path(input_files[0])
            output_path = self._validate_path(output_files[0])

            conn = None
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(query)
                results = cursor.fetchall()

                # Save results
                with open(output_path, 'w') as f:
                    json.dump({
                        'columns': [desc[0] for desc in cursor.description],
                        'rows': results
                    }, f, indent=2)

                return True
            finally:
                if conn:
                    conn.close()

        except Exception as e:
            logger.error(f"Error in B5: {str(e)}", exc_info=True)
            return False

    async def handle_b6(self, input_files: List[str], output_files: List[str], **kwargs):
        """Scrape website data"""
        try:
            url = kwargs.get('url')
            if not url:
                raise ValueError("URL is required")

            output_path = self._validate_path(output_files[0])

            # Fetch webpage
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"Web request failed: {response.status}")
                    html = await response.text()

            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')

            # Extract data
            data = {
                'title': soup.title.string if soup.title else None,
                'headings': [h.text for h in soup.find_all(['h1', 'h2', 'h3'])],
                'links': [a.get('href') for a in soup.find_all('a')],
                'text': soup.get_text()
            }

            # Save data
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error in B6: {str(e)}", exc_info=True)
            return False

    async def handle_b7(self, input_files: List[str], output_files: List[str], **kwargs):
        """Compress or resize image"""
        try:
            input_path = self._validate_path(input_files[0])
            output_path = self._validate_path(output_files[0])

            # Process image
            with Image.open(input_path) as img:
                # Get parameters
                width = kwargs.get('width', img.width // 2)
                height = kwargs.get('height', img.height // 2)
                quality = kwargs.get('quality', 85)

                # Resize if specified
                if width and height:
                    img = img.resize((width, height), Image.Resampling.LANCZOS)

                # Save with compression
                img.save(output_path, quality=quality, optimize=True)

            return True
        except Exception as e:
            logger.error(f"Error in B7: {str(e)}", exc_info=True)
            return False

    async def handle_b8(self, input_files: List[str], output_files: List[str], **kwargs):
        """Transcribe audio from MP3"""
        try:
            input_path = self._validate_path(input_files[0])
            output_path = self._validate_path(output_files[0])

            # Convert audio to base64
            with open(input_path, 'rb') as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')

            # Use LLM for transcription
            prompt = """Transcribe this audio file.
            Return only the transcribed text.
            
            [Audio content is base64 encoded in the next message]
            """

            transcription = await call_llm(prompt + "\n" + audio_base64)

            # Save transcription
            with open(output_path, 'w') as f:
                f.write(transcription)

            return True
        except Exception as e:
            logger.error(f"Error in B8: {str(e)}", exc_info=True)
            return False

    async def handle_b9(self, input_files: List[str], output_files: List[str], **kwargs):
        """Convert Markdown to HTML"""
        try:
            input_path = self._validate_path(input_files[0])
            output_path = self._validate_path(output_files[0])

            # Read markdown
            with open(input_path, 'r') as f:
                markdown_text = f.read()

            # Convert to HTML
            html = markdown2.markdown(
                markdown_text,
                extras=['fenced-code-blocks', 'tables', 'break-on-newline']
            )

            # Add CSS if specified
            css = kwargs.get('css', """
                body { max-width: 800px; margin: 0 auto; padding: 20px; }
                pre { background: #f0f0f0; padding: 10px; }
                code { font-family: monospace; }
            """)

            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>{css}</style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """

            # Save HTML
            with open(output_path, 'w') as f:
                f.write(full_html)

            return True
        except Exception as e:
            logger.error(f"Error in B9: {str(e)}", exc_info=True)
            return False

    def create_csv_endpoint(self, app: FastAPI, csv_path: str):
        """Create endpoint to filter CSV and return JSON"""
        try:
            # Validate CSV path
            full_path = self._validate_path(csv_path)
            
            @app.get("/filter-csv")
            async def filter_csv(
                field: str = None, 
                value: str = None, 
                limit: int = 100,
                offset: int = 0
            ):
                try:
                    # Read CSV
                    with open(full_path, 'r') as f:
                        reader = csv.DictReader(f)
                        data = list(reader)

                    # Apply filters if specified
                    filtered_data = data
                    if field and value:
                        filtered_data = [
                            row for row in data 
                            if str(row.get(field, '')).lower() == str(value).lower()
                        ]

                    # Apply pagination
                    total_count = len(filtered_data)
                    paginated_data = filtered_data[offset:offset + limit]

                    return {
                        "total": total_count,
                        "offset": offset,
                        "limit": limit,
                        "data": paginated_data
                    }

                except Exception as e:
                    logger.error(f"CSV filtering error: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing CSV: {str(e)}"
                    )

            # Return the endpoint function for registration
            return filter_csv

        except Exception as e:
            logger.error(f"Error creating CSV endpoint: {str(e)}")
            raise ValueError(f"Failed to create CSV endpoint: {str(e)}")