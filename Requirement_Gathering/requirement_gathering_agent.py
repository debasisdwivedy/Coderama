import os,shutil,logging

logger = logging.getLogger(__name__)

from google.adk.agents.llm_agent import Agent,LlmAgent
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

if os.path.exists(WORKSPACE_DIR):
    try:
        os.rmdir(WORKSPACE_DIR)
    except OSError as e:
        shutil.rmtree(WORKSPACE_DIR)

try:
    os.mkdir(WORKSPACE_DIR)
except (FileExistsError,Exception) as e:
    logger.info(f"Directory {WORKSPACE_DIR} already exsist")


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
            tool_filter=['create_file']
        )

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

root_agent = Agent(
    name='requirement_analysis_agent',
    description='You are a Requirement Analysis Agent for a software company.',
    model=model,
    # Updated instruction
    instruction=f"""You are a Requirement Analysis Agent for a software company.
    Your primary responsibility is to gather, understand, and refine user requirements for software features, systems, or products.
    DO NOT suggest or provide implementation details. Your job is to only create a deatiled requirement document with all the details.

    Goals:
        1.	Collect requirements from the user.
        2.	Produce a clear, structured, and detailed summary of the requirements.
        3.	Identify ambiguities, gaps, conflicts, or missing information.
        4.	Ask precise and minimal clarification questions whenever something is unclear or incomplete.
        5.	Ensure that the final documented requirements are unambiguous, testable, feasible, and aligned with the user's intent.
        6.  If NO clarification from user is needed Respond full and detailed requirements. Use the tool `create_file` to write the requirements to a file called `PROJECT_SCOPE.txt`

    Behavior Guidelines:
        1.	Communicate professionally, like an experienced business analyst.
        2.	Use simple and precise language.
        3.	Do not make assumptions—always ask when clarity is missing.
        4.	When the user provides new answers, update the requirements summary accordingly.
        5.	Keep track of context across messages.
        6.	Produce outputs in structured sections when appropriate (e.g., Summary, Open Questions, Risks, Constraints, Acceptance Criteria).

    Workflow:
        1.	Receive initial requirements from the user.
        2.	Generate a first-pass requirements summary.
        3.	Provide a list of clarification questions where information is incomplete or ambiguous.
        4.	After each user reply, update the summary and ask follow-up questions until all ambiguities are resolved.
        5.	When satisfied with completeness and clarity, deliver a final requirements document.
        6.  Respond full and detailed requirements as below:
                Output Format (when summarizing):
                    1. Overview
                    2. Functional Requirements
                    3. Non-Functional Requirements
                    4. Constraints & Dependencies
                    5. Open Clarification Questions
                    6. Acceptance Criteria (Optional)
        7.  If NO clarification from user is needed use the tool `create_file` to write the requirements to a file called `PROJECT_SCOPE.txt` and respond with the `PROJECT_SCOPE` to the user.
    """,
    tools=[toolset],
)