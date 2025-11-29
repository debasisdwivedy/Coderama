import os,logging

logger = logging.getLogger(__name__)

from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters


WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")
DOCKER_IMAGE_TAG = os.getenv("DOCKER_IMAGE_TAG")
PLATFORM = os.getenv("PLATFORM")
PROVIDER = os.getenv("PROVIDER")
MODEL = os.getenv("MODEL")
SANDBOX_AVAILABLE = os.getenv("SANDBOX_AVAILABLE")

if PROVIDER is None:
  raise ValueError("Please select either `LITELLM` or `GOOGLE` as a provider in .env file")

if PROVIDER.lower() == "litellm":
    if(os.getenv("LITE_LLM_TOKEN") is None or os.getenv("LITE_LLM_TOKEN") == ""):
        raise ValueError("Please provide `LITE_LLM_TOKEN` for the provider in .env file")
    else:
        model = LiteLlm(model=f"{MODEL}",api_key=os.getenv("LITE_LLM_TOKEN"),num_retries=2)


if PROVIDER.lower() == "google": 
    if(os.getenv("GOOGLE_API_KEY") is None or os.getenv("GOOGLE_API_KEY") == ""):
        raise ValueError("Please provide `GOOGLE_API_KEY` in .env file")
    else:
        model = f"{MODEL}"

toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params = StdioServerParameters(
                    command='coderama',
                    env={
                        "WORKSPACE_DIR":WORKSPACE_DIR,
                        "DOCKER_IMAGE_TAG":DOCKER_IMAGE_TAG,
                        "PLATFORM":PLATFORM,
                        "SANDBOX_AVAILABLE":SANDBOX_AVAILABLE,
                    },
                ),
            ),
            tool_filter=['create_file','read_file','create_folder']
        )

root_agent = Agent(
  name='Project_Manager',
  description='A Project Manager to critique and validate of the requirements provided by the user.',
  model=model,
  # Updated instruction
  instruction=f"""You are a Project Manager to make sure the application delivered is correct and with proper standards.
  DO NOT suggest or provide implementation details. Your job is to only create the `sprints` based on the requirements.
  Read the product requirement from a particular file called `PROJECT_SCOPE.txt` using the tool `read_file`
  Convert the requirements into `N` number of **SPRINTS** where N = The total number of Sprints and a Sprint is a 10 days work week.
    
  **Task:**
    a ) Read the requirement from file `PROJECT_SCOPE.txt` using tool `read_file`
    b) Analyze the 'Requirements'.
    c) Convert it into managable work and divide it into sprints for the team to work on and deliver on time.
    d) Create a folder called `sprints` using tool `create_folder` is not present.
    d) Write the GOALS as SPRINT-<NUMBER>.txt using the tool `create_file` in the `sprints` folder. 
    e) Respond to the user with ONLY the total NUMBER of SPRINT created. ASK the user if they want to proceed with the SPRINT.
  """,
  tools=[toolset],
  output_key="final_product_requirement"
)