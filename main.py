# from fastapi import FastAPI, HTTPException, Query, Form, Request
# from fastapi.responses import PlainTextResponse, Response
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# import os
# import sys
# import logging
# from dotenv import load_dotenv
# from task_handlers import TaskHandler
# from llm_helper import parse_task_description

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(title="DataWorks Task Automation")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Constants
# # DATA_DIR = (Path("data") if os.getenv("ENV") == "development" else Path("/data")).resolve()
# DATA_DIR = Path("/data").resolve()
# AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# if not AIPROXY_TOKEN:
#     logger.warning("AIPROXY_TOKEN not set. Some functionality may be limited.")

# # Create data directory
# DATA_DIR.mkdir(parents=True, exist_ok=True)
# logger.info(f"Using data directory: {DATA_DIR}")

# # Initialize task handler
# task_handler = TaskHandler(data_dir=DATA_DIR)

# def validate_path(path: str) -> Path:
#     """Basic path validation to ensure paths are within data directory"""
#     # Remove leading slash if present and 'data/' prefix
#     clean_path = path.lstrip('/')
#     if clean_path.startswith('data/'):
#         clean_path = clean_path[5:]
    
#     # Resolve the full path
#     full_path = (DATA_DIR / clean_path).resolve()
    
#     # Ensure the path is within DATA_DIR
#     if not str(full_path).startswith(str(DATA_DIR)):
#         raise ValueError(f"Path {path} is outside data directory")
        
#     return full_path

# @app.post("/run")
# @app.get("/run")
# async def run_task(request: Request, task: str = None):
#     """Execute a plain-English task using LLM for parsing and task execution."""
#     try:
#         # Get task from either query params or form data
#         if task is None:
#             if request.method == "POST":
#                 form_data = await request.form()
#                 task = form_data.get("task")
#             if task is None:
#                 raise HTTPException(
#                     status_code=400,
#                     detail="Task parameter is required"
#                 )

#         logger.info(f"Received task: {task}")

#         # Parse task using LLM
#         try:
#             parsed_task = await parse_task_description(task, query_url=str(request.url))
#             logger.info(f"Successfully parsed task: {parsed_task}")
#         except Exception as e:
#             logger.error(f"Failed to parse task: {e}")
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Failed to understand task: {str(e)}"
#             )
        
#         # Validate paths
#         try:
#             for file_path in parsed_task.get('input_files', []):
#                 validate_path(file_path)
#             for file_path in parsed_task.get('output_files', []):
#                 validate_path(file_path)
#         except Exception as e:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid file path: {str(e)}"
#             )
        
#         # Execute task
#         try:
#             # Remove original_task from parameters to avoid duplication
#             task_params = parsed_task.get('parameters', {}).copy()
#             if 'original_task' in task_params:
#                 del task_params['original_task']

#             result = await task_handler.dispatch_task(
#                 task_type=parsed_task['task_type'].upper(),
#                 input_files=parsed_task['input_files'],
#                 output_files=parsed_task['output_files'],
#                 app=app if parsed_task['task_type'].upper() == 'B10' else None,
#                 **task_params
#             )
            
#             if result:
#                 return {
#                     "status": "success",
#                     "message": f"Task {parsed_task['task_type']} completed successfully",
#                     "task_info": parsed_task
#                 }
#             else:
#                 raise HTTPException(
#                     status_code=500,
#                     detail=f"Task {parsed_task['task_type']} failed to execute"
#                 )
                
#         except Exception as e:
#             logger.error(f"Task execution failed: {e}", exc_info=True)
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to execute task: {str(e)}"
#             )
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Internal server error: {str(e)}"
#         )

# @app.get("/read/{path:path}", response_class=PlainTextResponse)
# async def read_file(path: str):
#     """Read and return the contents of a file within the data directory."""
#     try:
#         # Handle both /data/ prefix and no prefix
#         if path.startswith('/'):
#             path = path[1:]
#         if path.startswith('data/'):
#             path = path[5:]
            
#         # Validate path
#         try:
#             full_path = validate_path(path)
#         except Exception as e:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid path: {str(e)}"
#             )
        
#         # Check if file exists
#         if not full_path.exists():
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"File not found: {path}"
#             )
        
#         # Read file content
#         try:
#             with open(full_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#                 return content
#         except UnicodeDecodeError:
#             # Try binary read for non-text files
#             with open(full_path, 'rb') as f:
#                 content = f.read()
#                 return Response(content=content, media_type='application/octet-stream')
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error reading file: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error reading file: {str(e)}"
#         )
    
# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     return {
#         "status": "healthy",
#         "data_dir": str(DATA_DIR),
#         "data_dir_exists": DATA_DIR.exists(),
#         "data_dir_is_dir": DATA_DIR.is_dir() if DATA_DIR.exists() else False
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)












from fastapi import FastAPI, HTTPException, Query, Form, Request
from fastapi.responses import PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import sys
import logging
from dotenv import load_dotenv
from task_handlers import TaskHandler
from llm_helper import parse_task_description

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_data_directory() -> Path:
    """Set up the data directory exactly at /data as required by datagen.py and eval.py."""
    try:
        # Always try /data first as that's what datagen.py expects
        data_dir = Path("/data")
        
        # If /data is not accessible, create a symlink from a writable location
        if not os.access('/data', os.W_OK):
            # Create a temporary directory in user's home
            temp_data_dir = Path.home() / "dataworks_temp"
            temp_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to create symlink if /data doesn't exist
            try:
                if not os.path.exists('/data'):
                    os.symlink(str(temp_data_dir), '/data')
                    logger.info(f"Created symlink from /data to {temp_data_dir}")
                    data_dir = Path("/data")
                else:
                    # If /data exists but isn't writable, use temp directory directly
                    data_dir = temp_data_dir
                    logger.warning(f"Using alternative data directory: {data_dir}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not create symlink to /data: {e}")
                data_dir = temp_data_dir
                logger.warning(f"Using alternative data directory: {data_dir}")
        
        # Ensure the directory exists
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using data directory: {data_dir}")
        
        return data_dir
        
    except Exception as e:
        logger.error(f"Failed to set up data directory: {e}")
        raise RuntimeError(f"Cannot set up data directory: {e}")

# Initialize data directory
DATA_DIR = setup_data_directory()

# Check for AIPROXY_TOKEN
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    logger.warning("AIPROXY_TOKEN not set. Some functionality may be limited.")

# Initialize FastAPI app
app = FastAPI(title="DataWorks Task Automation",
            docs_url="/docs",
            redoc_url="/redoc")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize task handler
task_handler = TaskHandler(data_dir=DATA_DIR)

def validate_path(path: str) -> Path:
    """Basic path validation to ensure paths are within data directory"""
    # Remove leading slash if present and 'data/' prefix
    clean_path = path.lstrip('/')
    if clean_path.startswith('data/'):
        clean_path = clean_path[5:]
    
    # Resolve the full path
    full_path = (DATA_DIR / clean_path).resolve()
    
    # Ensure the path is within DATA_DIR
    if not str(full_path).startswith(str(DATA_DIR)):
        raise ValueError(f"Path {path} is outside data directory")
        
    return full_path

# @app.post("/run")
# @app.get("/run")
# async def run_task(request: Request, task: str = None):
#     """Execute a plain-English task using LLM for parsing and task execution."""
#     try:
#         # Get task from either query params or form data
#         if task is None:
#             if request.method == "POST":
#                 form_data = await request.form()
#                 task = form_data.get("task")
#             if task is None:
#                 raise HTTPException(
#                     status_code=400,
#                     detail="Task parameter is required"
#                 )

#         logger.info(f"Received task: {task}")

#         # Parse task using LLM
#         try:
#             parsed_task = await parse_task_description(task, query_url=str(request.url))
#             logger.info(f"Successfully parsed task: {parsed_task}")
#         except Exception as e:
#             logger.error(f"Failed to parse task: {e}")
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Failed to understand task: {str(e)}"
#             )
        
#         # Validate paths
#         try:
#             for file_path in parsed_task.get('input_files', []):
#                 validate_path(file_path)
#             for file_path in parsed_task.get('output_files', []):
#                 # Ensure output file's parent directory exists
#                 output_path = validate_path(file_path)
#                 output_path.parent.mkdir(parents=True, exist_ok=True)
#         except Exception as e:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid file path: {str(e)}"
#             )
        
#         # Execute task
#         try:
#             # Remove original_task from parameters to avoid duplication
#             task_params = parsed_task.get('parameters', {}).copy()
#             if 'original_task' in task_params:
#                 del task_params['original_task']

#             result = await task_handler.dispatch_task(
#                 task_type=parsed_task['task_type'].upper(),
#                 input_files=parsed_task['input_files'],
#                 output_files=parsed_task['output_files'],
#                 app=app if parsed_task['task_type'].upper() == 'B10' else None,
#                 **task_params
#             )
            
#             if result:
#                 return {
#                     "status": "success",
#                     "message": f"Task {parsed_task['task_type']} completed successfully",
#                     "task_info": parsed_task
#                 }
#             else:
#                 raise HTTPException(
#                     status_code=500,
#                     detail=f"Task {parsed_task['task_type']} failed to execute"
#                 )
                
#         except Exception as e:
#             logger.error(f"Task execution failed: {e}", exc_info=True)
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to execute task: {str(e)}"
#             )
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Internal server error: {str(e)}"
#         )



@app.post("/run")
@app.get("/run")
async def run_task(request: Request, task: str = None):
    """Execute a task with proper file syncing."""
    try:
        if task is None:
            if request.method == "POST":
                form_data = await request.form()
                task = form_data.get("task")
            if task is None:
                raise HTTPException(status_code=400, detail="Task parameter is required")

        # Parse task
        parsed_task = await parse_task_description(task, query_url=str(request.url))
        
        # For A2, ensure format.md is in sync before and after
        if parsed_task['task_type'].upper() == 'A2':
            task_handler.sync_file("format.md")
            
        # Execute task
        result = await task_handler.dispatch_task(
            task_type=parsed_task['task_type'],
            input_files=parsed_task['input_files'],
            output_files=parsed_task['output_files'],
            **parsed_task.get('parameters', {})
        )
        
        # Sync again after A2
        if parsed_task['task_type'].upper() == 'A2':
            task_handler.sync_file("format.md")
            
        if result:
            return {"status": "success", "message": "Task completed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Task failed to execute")
            
    except Exception as e:
        logger.error(f"Error executing task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/read/{path:path}", response_class=PlainTextResponse)
# async def read_file(path: str):
#     """Read and return the contents of a file within the data directory."""
#     try:
#         # Clean the path
#         clean_path = path.lstrip('/')
#         if clean_path.startswith('data/'):
#             clean_path = clean_path[5:]

#         # Try multiple possible locations
#         possible_paths = [
#             DATA_DIR / clean_path,  # Local data directory
#             Path("/data") / clean_path,  # Root data directory
#             Path(os.getcwd()) / "data" / clean_path  # Current directory
#         ]
        
#         file_found = False
#         for try_path in possible_paths:
#             try:
#                 full_path = try_path.resolve()
#                 if full_path.exists():
#                     file_found = True
#                     logger.info(f"Found file at: {full_path}")
#                     break
#             except:
#                 continue

#         if not file_found:
#             error_msg = f"File not found: {path}. Tried: {', '.join(str(p) for p in possible_paths)}"
#             logger.error(error_msg)
#             raise HTTPException(
#                 status_code=404,
#                 detail=error_msg
#             )
        
#         # Read file content
#         try:
#             with open(full_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#                 return content
#         except UnicodeDecodeError:
#             # Try binary read for non-text files
#             with open(full_path, 'rb') as f:
#                 content = f.read()
#                 return Response(content=content, media_type='application/octet-stream')
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error reading file: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error reading file: {str(e)}"
#         )

@app.get("/read")
async def read_file(path: str):
    """Read and return the contents of a file within the data directory."""
    try:
        # Clean the path
        clean_path = path.lstrip('/')
        if clean_path.startswith('data/'):
            clean_path = clean_path[5:]
            
        # Try multiple possible locations in order
        possible_paths = [
            Path('/data') / clean_path,  # First try /data
            DATA_DIR / clean_path,       # Then try our configured data directory
        ]
        
        for try_path in possible_paths:
            if try_path.exists():
                try:
                    with open(try_path, 'r', encoding='utf-8') as f:
                        return PlainTextResponse(f.read())
                except UnicodeDecodeError:
                    # For binary files
                    with open(try_path, 'rb') as f:
                        return Response(content=f.read(), media_type='application/octet-stream')
                        
        # If we get here, file wasn't found
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {path}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed directory information."""
    # Get all possible data directory locations
    possible_dirs = [
        Path("/data"),
        Path.home() / "data",
        Path(os.getcwd()) / "data",
    ]
    
    dir_info = {}
    for dir_path in possible_dirs:
        try:
            dir_info[str(dir_path)] = {
                "exists": dir_path.exists(),
                "is_dir": dir_path.is_dir() if dir_path.exists() else False,
                "writable": os.access(dir_path, os.W_OK) if dir_path.exists() else False,
                "contents": [str(f.name) for f in dir_path.glob("*")] if dir_path.exists() and dir_path.is_dir() else []
            }
        except Exception as e:
            dir_info[str(dir_path)] = {"error": str(e)}
    
    return {
        "status": "healthy",
        "current_data_dir": str(DATA_DIR),
        "all_data_directories": dir_info,
        "cwd": str(Path.cwd()),
        "home": str(Path.home())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





