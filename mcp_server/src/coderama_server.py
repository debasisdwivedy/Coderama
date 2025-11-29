import os,logging,asyncio

logger = logging.getLogger(__name__)

from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("Coderama")

@mcp.tool
async def create_folder(folder_path:str)->dict:
    """
    Create a folder at the specified location and return an execution status.

    Agent Tool Specification:
        Name: create_folder
        Description: Creates a directory or folder at the given path.
                     Returns a structured status dictionary with success or error details.
        Input Arguments:
            folder_path (str): Path where the folder or file should be created.
        Output:
            dict: {
                "STATUS": bool,       # True if creation succeeded, False otherwise
                "PATH": str            # Full path of the created file/folder
            }

    Behavior:
        - Validates the input path and constructs the full target path.
        - Creates any missing directories when creating a file.
        - Overwrites nothing; if the target already exists, marks the operation as successful.
        - Captures and reports exceptions in the status dictionary.

    """
    WORKSPACE_DIR=os.environ.get("WORKSPACE_DIR")
    folder_path = WORKSPACE_DIR+os.sep+folder_path
    if not safe_join(WORKSPACE_DIR,folder_path):
        return {
            "STATUS": "FAILURE",
            "PATH": f"{folder_path} outside working directory. Do not use eacape sequence in your directory path"
        }
    try:
        os.makedirs(folder_path,exist_ok=True)
    except (FileExistsError,Exception) as e:
        logger.error("File exsist {folder_path}")
        #print(f"File exsist {folder_path}",file=sys.stderr)
    return {
        "STATUS": "SUCCESS",
        "PATH": f"{folder_path}"
    }
        
@mcp.tool
async def create_file(parent_folder_path:str,file_name:str,content:str)->dict:
    """
    Create a file at the specified location and return an execution status.

    Agent Tool Specification:
        Name: create_file
        Description: Creates a file at the given path. Returns a structured status dictionary with success or error details.
        Input Arguments:
            parent_folder_path (str): Parent folder path where the file should be created.Can be empty if it is the root.
            file_name (str): The name of the file to create. Provide only the filename, with no directory paths or separators included.
            content (str): Content to be written to a file
        Output:
            dict: {
                "STATUS": bool,       # True if creation succeeded, False otherwise
                "PATH": str            # Full path of the created file/folder
            }

    Behavior:
        - Validates the input path and constructs the full target path.
        - Creates any missing directories when creating a file.
        - Overwrites nothing; if the target already exists, marks the operation as successful.
        - Captures and reports exceptions in the status dictionary.

    """
    WORKSPACE_DIR=os.environ.get("WORKSPACE_DIR")
    folder_path = WORKSPACE_DIR+os.sep+parent_folder_path
    try:
        os.makedirs(folder_path,exist_ok=True)
    except (FileExistsError,Exception) as e:
        logger.error(f"File exsist {folder_path}")
    full_path = folder_path+os.sep+file_name
    if not safe_join(WORKSPACE_DIR,full_path):
        return {
            "STATUS": "FAILURE",
            "PATH": f"{full_path} outside working directory. Do not use eacape sequence in your directory path"
        }
    try:
        with open(full_path,"w") as f:
            f.write(content)
            logger.info(f"file created {full_path}")
        
        return {
            "STATUS": "SUCCESS",
            "PATH": f"{full_path}"
        }
    except (FileNotFoundError,NotADirectoryError,IsADirectoryError,Exception) as e:
        return {
            "STATUS": "FAILURE",
            "PATH": f"{e}"
        }

@mcp.tool
async def read_file(folder_path:str,file_name:str)->dict:
    """
    Create a folder or file at the specified location and return an execution status.

    Agent Tool Specification:
        Name: read_file
        Description: Reads a file at a particular location and provides the content of the file.
        Input Arguments:
            folder_path (str): Relative path of the folder where file has to be read.
            file_name (str): Name of the file.
        Output:
            dict: {
                "STATUS": bool,       # True if creation succeeded, False otherwise
                "CONTENT": str            # Full path of the created file/folder
            }

    """
    WORKSPACE_DIR=os.environ.get("WORKSPACE_DIR")
    file_path = WORKSPACE_DIR+os.sep+folder_path+os.sep+file_name
    if not safe_join(WORKSPACE_DIR,file_path):
        return {
            "STATUS": "FAILURE",
            "PATH": f"{file_path} outside working directory. Do not use eacape sequence in your directory path"
        }
    if not os.path.exists(file_path):
        return {
        "STATUS": "FAILURE",
        "CONTENT": f"File does not exsist at {file_path}"
    }
    with open(file_path,"r") as f:
        content = f.read()
    return {
        "STATUS": "SUCCESS",
        "CONTENT": content
    }

@mcp.tool
async def execute_shell_code(shell_file_path:str)->dict:
    """
    Execute a shell code file located at the given path in shell and return an execution status object.

    Parameters
    ----------
    shell_file_path (str): The name of the shell script file to execute. 
                    You must provide ONLY the filename of an existing shell script, not a raw shell command. 
                    Do NOT include inline shell commands, arguments, pipes, flags, or bash syntax. 
                    Do NOT include any directory paths — provide only the script filename (e.g., "deploy.sh"), 
                    which will be executed from the workspace directory.

    Returns
    -------
    dict
        A structured status response with the following keys:
            - "STATUS" (bool): Whether execution completed without errors.
            - "OUTPUT" (str): Captured standard output as text.
            - "ERROR" (str): Captured standard error as text.
            - "REASON" (str): A human-readable summary of the result.

    Notes
    -----
    - This function does not return live streams; it captures output after execution.
    - Errors during file loading or process start are returned as part of the
      structured status rather than raised as exceptions.
    """
    import subprocess
    WORKSPACE_DIR=os.environ.get("WORKSPACE_DIR")
    DOCKER_IMAGE_TAG = os.environ.get("DOCKER_IMAGE_TAG")
    PLATFORM = os.environ.get("PLATFORM")
    SANDBOX_AVAILABLE = os.environ.get("SANDBOX_AVAILABLE")
    logger.info(f"""
                ========================IN EXECUTE SHELL CODE============================
                    The WORKSPACE_DIR is {WORKSPACE_DIR}
                    DOCKER_IMAGE_TAG is {DOCKER_IMAGE_TAG}
                    PLATFORM is {PLATFORM}
                    SANDBOX_AVAILABLE is {SANDBOX_AVAILABLE}
    """)
    if not shell_file_path.endswith(".sh"):
        return {
            "STATUS": "FAILURE",
            "REASON": "PLEASE CREATE A SHELL SCRIPT FILE e.g., script.sh to execute shell commands.CANNOT EXECUTE COMMAND DIRECTLY.",
            "WORKSPACE": f"{shell_file_path}"
        }

    try:
        logger.info("   =====BEFORE EXECUTE SUBPROCESS===============")
        if SANDBOX_AVAILABLE is not None and SANDBOX_AVAILABLE.lower()=="true":
            logger.info("   ===== SANDBOX IS SET===============")
            if PLATFORM is not None and PLATFORM == "linux/arm64":
                cmd = [
                "docker", "run",
                f"--platform={PLATFORM}",
                "--rm",
                "-v", f"{WORKSPACE_DIR}:/app",
                "-w", "/app",
                f"{DOCKER_IMAGE_TAG}",
                "bash", "-c",
                f"chmod +x {shell_file_path} && ./{shell_file_path}"
            ]
            else:
                cmd = [
                "docker", "run",
                "--rm",
                "-v", f"{WORKSPACE_DIR}:/app",
                "-w", "/app",
                f"{DOCKER_IMAGE_TAG}",
                "bash", "-c",
                f"chmod +x {shell_file_path} && ./{shell_file_path}"
            ]
            logger.info(f"   Command to be executed : {cmd}")
            proc = await asyncio.create_subprocess_exec(*cmd,
                                                        stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE,
                                                        stdin=asyncio.subprocess.DEVNULL
                                                        )
            out, err = await proc.communicate()
        else:
            logger.info("   ===== SANDBOX IS UNSET===============")
            script_path = os.path.join(WORKSPACE_DIR, shell_file_path)
            cmd = ["bash", "-c", f"chmod +x {script_path} && ./{shell_file_path}"]
            logger.info(f"   Command to be executed : {cmd}")
            proc = await asyncio.create_subprocess_exec(*cmd,
                                                        stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE,
                                                        stdin=asyncio.subprocess.DEVNULL,
                                                        cwd=WORKSPACE_DIR
                                                        )
            out, err = await proc.communicate()
        logger.info("   =====AFTER EXECUTE SUBPROCESS===============")
        logger.info(f"Output: {out.decode() if out else ''}")
        logger.info(f"Errors: {err.decode() if err else ''}")
        return {
            "STATUS": "SUCCESS",
            "REASON": "CODE EXECUTED SUCCESSFULLY",
            "OUTPUT": f"{out.decode() if out else ''}",
            "ERROR": f"{err.decode() if err else ''}",
            "COMMAND": f"{' '.join(cmd)}"
        }
    except (subprocess.CalledProcessError,Exception) as e:
        logger.error(f"Command failed with exit code {e.returncode}: {e.stderr} : {e.stdout}")
        return {
            "STATUS": "FAILURE",
            "OUTPUT": f"{e.stdout}",
            "ERROR": f"{e.stderr}",
            "RETURN_CODE": f"CODE EXECUTION FAILE WITH ERROR {e.returncode}",
            "COMMAND": f"{' '.join(cmd)}"
        }

@mcp.tool
async def web_search(query:str,max_results:int=20) -> dict:
    """
    Tool: Web Search

        Name : web_search

        Description:
            This tool performs a search operation for a given user query.
            It should be invoked whenever a user makes a request, asks a question, or provides any input that requires information retrieval.
            Always use this tool to fetch relevant information before attempting to answer the user's query.

        Args:
            query:str = The user's query or question that needs to be searched.
            max_results:int = (OPTIONAL) The total number of results/links returned from the web search. DEFAULT value is 20. START with 5 and keep INCREASING the value till 20 if the results are unsatisfactory.

        Usage:
            Call this tool immediately upon receiving a user query to ensure that the LLM has the most accurate and updated context for generating a response.

        Output:
            A dictionary of results
    """
    from ddgs import DDGS
    from googlesearch import search
    search_results = {}

    ddgs = DDGS(verify=False,timeout=5)
    results = ddgs.text(query, safesearch='off',max_results=max_results)
    # for result in results:
    #     print(result)

    if len(results) > 0:
        search_results["Duck_Duck_GO"]=results

    google_results=[]
    for result in search(query, num_results=max_results,unique=True,advanced=True,ssl_verify=True,sleep_interval=5):
        google_results.append(result)
    
    if len(google_results) > 0:
        search_results["Google"]=google_results

    if len(search_results) > 0:
        return search_results
    else:
        return {
            "result": "Unable todo a web search right now. Please try again later!!!!!"
        }
    

def safe_join(workspace: str, user_path: str) -> bool:
    workspace_path = Path(workspace).resolve()               # absolute path
    target_path = (workspace_path / user_path).resolve()     # resolve user input
    
    # Check if the final path is inside the workspace
    if target_path.is_symlink() or not target_path.relative_to(workspace) or not str(target_path).startswith(str(workspace_path)):
        return False
    
    return True

async def run_server():
    logger.info(" ==================Starting MCP server ==================")
    await mcp.run_stdio_async()
