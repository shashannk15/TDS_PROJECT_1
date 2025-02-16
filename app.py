from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, Callable
import subprocess
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
from datetime import datetime
from dateutil.parser import parse
import glob
import numpy as np
import requests
from fastapi.middleware.cors import CORSMiddleware



# Load environment variables
load_dotenv()

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
OPEN_AI_PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1"

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=['GET','POST'],
    allow_headers=["*"]
)

# Initialize Groq by OpenAI client
# client = OpenAI(
#     base_url=os.environ.get("OPEN_AI_PROXY_URL"),
#     api_key=os.environ.get("AIPROXY_TOKEN"),
# )

client = OpenAI(
    base_url=OPEN_AI_PROXY_URL,
    api_key=AIPROXY_TOKEN,
)

# Constants
RUNNING_IN_CODESPACES = "CODESPACES" in os.environ
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")

def ensure_local_path(path: str) -> str:
    """Ensure the path uses local format"""
    if ((not RUNNING_IN_CODESPACES) and RUNNING_IN_DOCKER):
        return path
    else:
        return path.lstrip("/")

def download_file(url: str, local_filename: str) -> str:
    """Download a file from URL"""
    response = requests.get(url)
    response.raise_for_status()
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    return local_filename

def install_and_run_script(package: str, script_url: str, args: list):
    """Install a package and run a script from URL with arguments"""
    try:
        # Install package if not already installed
        subprocess.run(["pip", "install", package], check=True)
        
        # Download the script
        script_name = script_url.split("/")[-1]
        download_file(script_url, script_name)
        
        # Make sure the script is executable
        os.chmod(script_name, 0o755)
        
        # Run the script with uv
        subprocess.run(["uv", "run", script_name, "--root", "./data"] + args, check=True)
        
        return {"status": "success", "message": "Script executed successfully"}
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error executing command: {str(e)}")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

# Additional task functions

def count_weekdays(input_file: str, output_file: str, weekday: str):
    """Count occurrences of a specific weekday in dates file"""
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    
    weekdays = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    
    count = 0
    with open(input_file_path, 'r') as f:
        for line in f:
            try:
                date = parse(line.strip())
                if date.weekday() == weekdays[weekday]:
                    count += 1
            except ValueError:
                continue
    
    with open(output_file_path, 'w') as f:
        f.write(str(count))

def create_markdown_index(docs_dir: str, output_file: str):
    """Create index of markdown H1 headers"""
    docs_dir_path = ensure_local_path(docs_dir)
    output_file_path = ensure_local_path(output_file)
    
    index = {}
    for root, _, files in os.walk(docs_dir_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, docs_dir_path)
                
                with open(file_path, 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if line.startswith('# '):
                            index[relative_path] = line.lstrip('# ').strip()
                            break
    
    with open(output_file_path, 'w') as f:
        json.dump(index, f, indent=2)

def extract_email_address(email_content: str, output_file: str):
    """Extract email address using LLM"""
    email_path = ensure_local_path(email_content)
    output_path = ensure_local_path(output_file)
    
    with open(email_path, 'r') as f:
        content = f.read()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the sender's email address from the following email. Respond with only the email address."},
            {"role": "user", "content": content}
        ]
    )
    
    email = response.choices[0].message.content.strip()
    with open(output_path, 'w') as f:
        f.write(email)

def extract_credit_card_number(image_path: str, output_file: str):
    """Extract credit card number from image using LLM"""
    import base64
    
    image_path = ensure_local_path(image_path)
    output_path = ensure_local_path(output_file)
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the credit card number from this image. Return only the number without spaces."},
                    {"type": "image_path", "image_path": {"path": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    )
    
    card_number = response.choices[0].message.content.strip().replace(" ", "")
    with open(output_path, 'w') as f:
        f.write(card_number)

def find_similar_comments(comments_file: str, output_file: str):
    """Find most similar pair of comments using embeddings"""
    from sentence_transformers import SentenceTransformer
    
    comments_path = ensure_local_path(comments_file)
    output_path = ensure_local_path(output_file)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    with open(comments_path, 'r') as f:
        comments = [line.strip() for line in f.readlines()]
    
    embeddings = model.encode(comments)
    similarity_matrix = np.inner(embeddings, embeddings)
    np.fill_diagonal(similarity_matrix, -1)
    
    max_idx = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
    similar_pair = [comments[max_idx[0]], comments[max_idx[1]]]
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(similar_pair))

def calculate_ticket_sales(db_file: str, ticket_type: str, output_file: str):
    """Calculate total sales for specific ticket type"""
    db_path = ensure_local_path(db_file)
    output_path = ensure_local_path(output_file)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT SUM(units * price)
        FROM tickets
        WHERE type = ?
    """, (ticket_type,))
    
    total = cursor.fetchone()[0] or 0
    conn.close()
    
    with open(output_path, 'w') as f:
        f.write(str(total))        

def format_file_with_prettier(file_path: str, prettier_version: str):
    """Format a file using Prettier"""
    input_file_path = ensure_local_path(file_path)
    subprocess.run(["npx", f"prettier@{prettier_version}", "--write", input_file_path], check=True)

def query_database(query: str, database_path: str):
    """Execute a database query"""
    db_path = ensure_local_path(database_path)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    finally:
        conn.close()

def sort_json_by_keys(input_file: str, output_file: str, keys: list):
    """Sort JSON data by specified keys"""
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    
    with open(input_file_path, "r") as file:
        data = json.load(file)
    
    sorted_data = sorted(data, key=lambda x: tuple(x[key] for key in keys))
    
    with open(output_file_path, "w") as file:
        json.dump(sorted_data, file, indent=2)

def process_and_write_logfiles(logs_dir: str, output_file: str, num_logs: int = 10):
    """Process recent log files and write their first lines"""
    logs_dir_path = ensure_local_path(logs_dir)
    output_file_path = ensure_local_path(output_file)
    
    log_files = glob.glob(os.path.join(logs_dir_path, "*.log"))
    log_files.sort(key=os.path.getmtime, reverse=True)
    recent_logs = log_files[:num_logs]
    
    with open(output_file_path, "w") as outfile:
        for log_file in recent_logs:
            with open(log_file, "r") as infile:
                first_line = infile.readline().strip()
                outfile.write(f"{first_line}\n")


# Additional utility functions
def validate_path(path: str) -> bool:
    """Validate that path is within /data directory"""
    abs_path = os.path.abspath(path)
    return abs_path.startswith("/data") or abs_path.startswith("./data")

def safe_file_operation(func):
    """Decorator to ensure safe file operations"""
    def wrapper(*args, **kwargs):
        # Validate all string arguments as potential paths
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, str) and ('/' in arg or '\\' in arg):
                path = ensure_local_path(arg)
                if not validate_path(path):
                    raise ValueError(f"Access denied to path outside /data: {arg}")
        return func(*args, **kwargs)
    return wrapper

# Business task functions
def fetch_api_data(api_url: str, output_file: str, headers: dict = None):
    """Task B3: Fetch data from API and save it"""
    output_path = ensure_local_path(output_file)
    
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'w') as f:
            json.dump(response.json(), f, indent=2)
        return {"status": "success", "message": "API data fetched and saved"}
    except Exception as e:
        raise Exception(f"Error fetching API data: {str(e)}")

def handle_git_operations(repo_url: str, target_dir: str, commit_message: str):
    """Task B4: Clone repo and make commit"""
    import git
    
    target_path = ensure_local_path(target_dir)
    if not validate_path(target_path):
        raise ValueError("Invalid target directory")
    
    try:
        # Clone repository
        repo = git.Repo.clone_from(repo_url, target_path)
        
        # Make changes (example)
        repo.index.add('*')
        repo.index.commit(commit_message)
        
        return {"status": "success", "message": "Git operations completed"}
    except Exception as e:
        raise Exception(f"Error in git operations: {str(e)}")

def run_database_query(query: str, db_file: str, output_file: str):
    """Task B5: Run SQL query and save results"""
    import pandas as pd
    
    db_path = ensure_local_path(db_file)
    output_path = ensure_local_path(output_file)
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df.to_json(output_path, orient='records', indent=2)
        return {"status": "success", "message": "Query executed and results saved"}
    except Exception as e:
        raise Exception(f"Error executing query: {str(e)}")

def scrape_website(url: str, output_file: str):
    """Task B6: Scrape website data"""
    from bs4 import BeautifulSoup
    
    output_path = ensure_local_path(output_file)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        data = {
            'title': soup.title.string if soup.title else None,
            'text': soup.get_text(),
            'links': [{'text': a.text, 'href': a.get('href')} for a in soup.find_all('a', href=True)]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        return {"status": "success", "message": "Website scraped successfully"}
    except Exception as e:
        raise Exception(f"Error scraping website: {str(e)}")

def process_image(image_path: str, output_path: str, max_size: tuple):
    """Task B7: Process image (resize/compress)"""
    from PIL import Image
    
    input_path = ensure_local_path(image_path)
    output_path = ensure_local_path(output_path)
    
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Resize maintaining aspect ratio
            img.thumbnail(max_size)
            
            # Save with optimization
            img.save(output_path, 'JPEG', quality=85, optimize=True)
        return {"status": "success", "message": "Image processed successfully"}
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def transcribe_audio(audio_file: str, output_file: str):
    """Task B8: Transcribe audio file"""
    input_path = ensure_local_path(audio_file)
    output_path = ensure_local_path(output_file)
    
    try:
        with open(input_path, "rb") as audio:
            response = client.audio.transcriptions.create(
                model="gpt-4o-mini",
                file=audio
            )
        
        with open(output_path, 'w') as f:
            f.write(response.text)
        return {"status": "success", "message": "Audio transcribed successfully"}
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")

def convert_markdown_to_html(markdown_file: str, output_file: str):
    """Task B9: Convert Markdown to HTML"""
    import markdown
    
    input_path = ensure_local_path(markdown_file)
    output_path = ensure_local_path(output_file)
    
    try:
        with open(input_path, 'r') as f:
            md_content = f.read()
        
        # Convert with extra features
        html = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
        
        # Add basic CSS
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 1em; }}
                code {{ background: #f4f4f4; padding: 2px 5px; }}
                pre {{ background: #f4f4f4; padding: 1em; overflow-x: auto; }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_doc)
        return {"status": "success", "message": "Markdown converted to HTML"}
    except Exception as e:
        raise Exception(f"Error converting markdown: {str(e)}")

# Function mappings
function_mappings = {
    "install_and_run_script": install_and_run_script,
    "format_file_with_prettier": format_file_with_prettier,
    "query_database": query_database,
    "sort_json_by_keys": sort_json_by_keys,
    "process_and_write_logfiles": process_and_write_logfiles,
    "count_weekdays": count_weekdays,
    "create_markdown_index": create_markdown_index,
    "extract_email_address": extract_email_address,
    "extract_credit_card_number": extract_credit_card_number,
    "find_similar_comments": find_similar_comments,
    "calculate_ticket_sales": calculate_ticket_sales,
    "fetch_api_data": fetch_api_data,
    "handle_git_operations": handle_git_operations,
    "run_database_query": run_database_query,
    "scrape_website": scrape_website,
    "process_image": process_image,
    "transcribe_audio": transcribe_audio,
    "convert_markdown_to_html": convert_markdown_to_html
}

# Tool definitions for OpenAI API
tools = [
    {
        "type": "function",
        "function": {
            "name": "install_and_run_script",
            "description": "Install a package and run a script with arguments (Task A1)",
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Package to install (e.g., 'uv')"
                    },
                    "script_url": {
                        "type": "string",
                        "description": "URL of the script to download and run"
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments to pass to the script (e.g., email address)"
                    }
                },
                "required": ["package", "script_url", "args"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_file_with_prettier",
            "description": "Format a file using Prettier with specific version (Task A2)",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to format"
                    },
                    "prettier_version": {
                        "type": "string",
                        "description": "Version of Prettier to use (e.g., '3.4.2')"
                    }
                },
                "required": ["file_path", "prettier_version"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_weekdays",
            "description": "Count occurrences of a specific weekday in a dates file (Task A3)",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to input file containing dates"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file for writing the count"
                    },
                    "weekday": {
                        "type": "string",
                        "description": "Weekday to count (e.g., 'Wednesday')",
                        "enum": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    }
                },
                "required": ["input_file", "output_file", "weekday"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_json_by_keys",
            "description": "Sort JSON array by specified keys (Task A4)",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "Path to input JSON file"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output JSON file"
                    },
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keys to sort by (e.g., ['last_name', 'first_name'])"
                    }
                },
                "required": ["input_file", "output_file", "keys"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_and_write_logfiles",
            "description": "Process recent log files and write first lines (Task A5)",
            "parameters": {
                "type": "object",
                "properties": {
                    "logs_dir": {
                        "type": "string",
                        "description": "Directory containing log files"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file"
                    },
                    "num_logs": {
                        "type": "integer",
                        "description": "Number of most recent logs to process",
                        "default": 10
                    }
                },
                "required": ["logs_dir", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_markdown_index",
            "description": "Create index of H1 headers from markdown files (Task A6)",
            "parameters": {
                "type": "object",
                "properties": {
                    "docs_dir": {
                        "type": "string",
                        "description": "Directory containing markdown files"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output index JSON file"
                    }
                },
                "required": ["docs_dir", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email_address",
            "description": "Extract sender's email address using LLM (Task A7)",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_content": {
                        "type": "string",
                        "description": "Path to file containing email content"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file for email address"
                    }
                },
                "required": ["email_content", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_credit_card_number",
            "description": "Extract credit card number from image using LLM (Task A8)",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image containing credit card"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file for card number"
                    }
                },
                "required": ["image_path", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_comments",
            "description": "Find most similar pair of comments using embeddings (Task A9)",
            "parameters": {
                "type": "object",
                "properties": {
                    "comments_file": {
                        "type": "string",
                        "description": "Path to file containing comments"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file for similar pair"
                    }
                },
                "required": ["comments_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_ticket_sales",
            "description": "Calculate total sales for specific ticket type (Task A10)",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_file": {
                        "type": "string",
                        "description": "Path to SQLite database file"
                    },
                    "ticket_type": {
                        "type": "string",
                        "description": "Ticket type to calculate sales for"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file for total sales"
                    }
                },
                "required": ["db_file", "ticket_type", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_api_data",
            "description": "Fetch data from an API and save it (Task B3)",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_url": {
                        "type": "string",
                        "description": "URL of the API endpoint"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save the API response"
                    }
                },
                "required": ["api_url", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_git_operations",
            "description": "Clone a git repo and make a commit (Task B4)",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": "URL of the git repository"
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Commit message"
                    }
                },
                "required": ["repo_url", "commit_message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_database_query",
            "description": "Run SQL query on database (Task B5)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "db_file": {
                        "type": "string",
                        "description": "Path to database file"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save query results"
                    }
                },
                "required": ["query", "db_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Extract data from a website (Task B6)",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the website to scrape"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save scraped data"
                    }
                },
                "required": ["url", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_image",
            "description": "Compress or resize an image (Task B7)",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to input image"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save processed image"
                    },
                    "max_size": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Maximum dimensions (width, height)",
                        "minItems": 2,
                        "maxItems": 2
                    }
                },
                "required": ["image_path", "output_path", "max_size"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": "Transcribe audio from MP3 file (Task B8)",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_file": {
                        "type": "string",
                        "description": "Path to MP3 file"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save transcription"
                    }
                },
                "required": ["audio_file", "output_file"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_markdown_to_html",
            "description": "Convert Markdown to HTML (Task B9)",
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown_file": {
                        "type": "string",
                        "description": "Path to markdown file"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to save HTML output"
                    }
                },
                "required": ["markdown_file", "output_file"]
            }
        }
    }
]

# @app.post("/run")
# async def run_task(task: str = Query(..., description="Task description")):
#     try:
#         # Log task receipt
#         print(f"Received task: {task}")
        
#         # Validate paths in task description
#         if not all(validate_path(path) for path in extract_paths_from_task(task)):
#             raise HTTPException(
#                 status_code=400,
#                 detail="Task contains invalid file paths outside /data directory"
#             )
        
#         # Call OpenAI API
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": """You are a task execution assistant. Parse the task description 
#                     and call the appropriate function. Ensure all file operations are within 
#                     the /data directory. Never delete files."""
#                 },
#                 {"role": "user", "content": task}
#             ],
#             tools=tools
#         )
        
#         # Log OpenAI response
#         print(f"OpenAI response: {response}")
        
#         results = []
#         if response.choices[0].message.tool_calls:
#             for tool_call in response.choices[0].message.tool_calls:
#                 function_name = tool_call.function.name
#                 function_args = json.loads(tool_call.function.arguments)
                
#                 print(f"Executing {function_name} with args: {function_args}")
                
#                 if function_to_call := function_mappings.get(function_name):
#                     try:
#                         result = function_to_call(**function_args)
#                         results.append({
#                             "function": function_name,
#                             "status": "success",
#                             "result": result
#                         })
#                     except Exception as e:
#                         results.append({
#                             "function": function_name,
#                             "status": "error",
#                             "error": str(e)
#                         })
#                 else:
#                     raise HTTPException(
#                         status_code=400,
#                         detail=f"Function {function_name} not implemented"
#                     )
        
#         return {
#             "status": "success",
#             "message": "Task execution completed",
#             "results": results
#         }
    
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/run")
async def run_task(task: str = Query(..., description="Plain-English task description")):
    """
    Execute a task based on natural language description
    """
    try:
        print(f"Received task: {task}")  # Debug logging
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                'role': 'system',
                'content': 'You are a helpful assistant that executes tasks based on natural language descriptions.'
            }, {
                'role': 'user',
                'content': task
            }],
            tools=tools
        )
        
        print(f"OpenAI response: {response}")  # Debug logging
        
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Executing function: {function_name}")  # Debug logging
                print(f"With arguments: {function_args}")  # Debug logging
                
                if function_to_call := function_mappings.get(function_name):
                    result = function_to_call(**function_args)
                    print(f"Function result: {result}")  # Debug logging
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Function {function_name} not found"
                    )
                    
        return {
            "status": "success",
            "message": "Task executed successfully"
        }
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")  # Debug logging
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="Path to the file to read")):
    """
    Read and return the contents of a file
    """
    output_file_path = ensure_local_path(path)
    if not os.path.exists(output_file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(output_file_path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file: {str(e)}"
        )
    
@app.get("/")
def home():
    return {"yayyyyy!!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)