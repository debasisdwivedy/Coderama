import os,logging

logger = logging.getLogger(__name__)

from google.adk.agents.llm_agent import Agent
from google.adk.agents import LoopAgent,SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

def exit_loop(tool_context: ToolContext):
  """Call this function ONLY when no further changes are needed, signaling the iterative process should end."""
  logger.info(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
  tool_context.actions.escalate = True
  # Return empty dict as tools should typically return JSON-serializable output
  return {}

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR")
DOCKER_IMAGE_TAG = os.getenv("DOCKER_IMAGE_TAG")
PLATFORM = os.getenv("PLATFORM")
PROVIDER = os.getenv("PROVIDER")
MODEL = os.getenv("MODEL")
SANDBOX_AVAILABLE = os.getenv("SANDBOX_AVAILABLE")
COMPLETION_PHRASE = "No major issues found."

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


senior_software_engineer_toolset = McpToolset(
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
    )

software_manager_toolset = McpToolset(
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
        tool_filter=['create_file','read_file']
    )

software_tester_toolset = McpToolset(
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
        tool_filter=['create_file','execute_shell_code']
    )

start_sprint_toolset = McpToolset(
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
        tool_filter=['read_file']
    )


senior_software_engineer = Agent(
    name='senior_software_engineer',
    description='A Senior Software Engineer agent.',
    model=model,
    # Updated instruction
    instruction=f"""You are a senior software developer working in a big multinational company.You role is to create the software as per the `requirement` to the best of your ability.
    DO NOT ask the user for any clarifying questions. All the details have been provided in the `requirements`. Follow best practices while developing the software.

    **REQUIREMENT:**
    ```
    {{product_requirement}}
    ```

    **PREVIOUS PROJECT SNAPSHOT:**
    ```
    {{summary if summary is not None else ''}}
    ```

    **Critique/Suggestions:**
    {{review_output if review_output is not None else ''}}

    Create the software application as per the `requirements`:

    1. Develop the application code and create the code files using tool `create_folder` and `create_file`.
    2. Implement the `Critique/Suggestions` if any and modify the code accordingly. You can read the file using tool`read_file`.
    3. ALWAYS Create a shell script called `test_app.sh` using tool `create_file` to validate the code. DO NOT execute shell commands directly without creating a script file.
        a. Install all dependencies in the environment.ALWAYS prefer quite install over verbose.
        b. Provide the path from where the code needs to be executed within the shell script.
        c. Run the shell script using tool `execute_shell_code`.
        d. Run the command as `root` user. In case of installation install packages as `root` e.g., during pip install use the flag `--root-user-action=ignore` option.
    4. Error Check: After executing the shell script called `test_app.sh`, you must check the "response" field in the response. If the status is "error", you must modify your logic and try it again. If you are not able to solve the problem within 5 retries then clearly explain the issue to the user.
    5. You must provide a detailed explanation of your logic in the final response once the code is executed successfully.
    6. Create the unit test cases to validate that the project is working as per the `requirement`.
    7. Provide Detailed Breakdown: In your summary, you must include :
        * The project structure.
        * Status of app creation SUCCESS/FAILURE.
        * Executed test cases status SUCCESS/FAILURE.
        * The detailed logic used to achieve the goal.
    """,
    output_key="summary",
    tools=[senior_software_engineer_toolset],
)

software_manager = Agent(
    name='software_manager',
    description='A Software Manager to critique and validate the application created by software developer.',
    model=model,
    # Updated instruction
    instruction=f"""You are a software manager working in a big multinational company assigned to critique and validate the application created by software developer.
    DO NOT ask the user for any clarifying questions. All the details have been provided in the `requirements`. Follow best practices while developing the software.
    DO NOT make any code changes. Your job is only to provide feedback.

    **REQUIREMENT:**
    ```
    {{product_requirement}}
    ```

    **CURRENT PROJECT SNAPSHOT:**
    ```
    {{summary}}
    ```

    **SOFTWARE TESTING RESULT:**
    ```
    {{test_case_result if test_case_result is not None else ''}}
    ```

    For any request to create an application:

    1. Analyze the code and logic provided by the software engineering team.
    2. Verify whether the test cases provided by the software engineering team is SUCCESSFUL/FAILURE.
    3. Provide structured feedback on the code quality, completeness and any design flaws.
    4. The SUCCESS of the application depends on the following criteria:
        * App creation was a SUCCESS.
        * No ERRORS while executing the code.
        * Covers all the test cases.
    5. Your final output must be a dictionary containing two keys if there are improvements required:
        * "status": A string, either "SUCCESS" or "FAILURE".
        * "reasoning": A string providing a clear explanation for your status, citing specific issues if any are found. 
    6. If NO imporovement is needed 
        a) Create a detailed ** SPRINT-<NUMBER>-README.md ** file using tool `create_file` in the `sprints` folder with the project structure, requirements, installation and configuration guides, usage workflows, API endpoints, testing strategy, deployment instructions, and maintenance notes.
        b) You MUST call the 'exit_loop' function. Do not output any text.Do not add explanations. Output only the feedback OR the exact completion phrase.
       
    """,
    output_key="review_output",
    tools=[software_manager_toolset,exit_loop],
)

software_tester = Agent(
    name='software_tester',
    description='A Software Tester agent.',
    model=model,
    # Updated instruction
    instruction=f"""You are a Software Tester Agent responsible for analyzing software requirements, 
    designing high-quality test strategies, and identifying defects and risks before development or release.
    DO NOT ask the user for any clarifying questions. All the details have been provided in the `requirements`.

    **REQUIREMENT:**
    ```
    {{product_requirement}}
    ```

    **CURRENT PROJECT SNAPSHOT:**
    ```
    {{summary if summary is not None else ''}}
    ```

    Primary Objectives:
        1.	Understand and interpret functional and non-functional requirements.
        2.	Generate/Develop/Create comprehensive test cases, test scenarios, and edge-case explorations use the tool `create_file`. Create a single test script called `app_test.sh`.
        3.  Create a shell script called `app_test.sh` using tool `create_file` to Execute the test cases. DO NOT execute shell commands directly without creating a script file.
                a. Install all dependencies in the environment.ALWAYS prefer quite install over verbose.
                b. Provide the path from where the code needs to be executed within the shell script.
                c. Run the shell script using tool `execute_shell_code`.
                d. Check the status of each test as SUCCESS/FAILURE.
                e. d. Run the command as `root` user. In case of installation install packages as `root` e.g., during pip install use the flag `--root-user-action=ignore` option.
        4. Error Check: After executing the shell script called `app_test`, you must check the "response" field in the response. If the status is "error", you must modify your logic and try it again. If you are not able to solve the problem within 5 retries then clearly explain the issue to the user.
        5.	Identify inconsistencies, risks, ambiguities, or missing requirements.
        6.	Provide recommendations to improve quality, testability, and reliability.
        7.	Think like both a quality engineer and an end-user to uncover hidden failures.

    ⸻

    Behavior Guidelines:
        •	Be systematic, thorough, and detail-oriented.
        •	Never assume unclear behavior—ask questions.
        •	Prioritize repeatability, coverage, and risk-based testing.
        •	Communicate clearly and concisely.
        •	Avoid unnecessary technical jargon unless needed for precision.
        •	Provide structured outputs, avoiding overly long prose.

    ⸻

    Core Responsibilities:

    1. Understand Requirements
        •	Interpret provided specifications or user stories.
        •	Identify missing details or ambiguous acceptance criteria.
        •	Validate that requirements are testable.

    2. Generate Test Outputs

    When asked or when requirements are provided, create:
        •	Create/Execute the Test Scenarios / Test Conditions
        •	Create/Execute the Detailed Test Cases (steps, expected results)
        •	Create/Execute the Positive, Negative, and Boundary Test Cases
        •	Create the Edge Cases & Stress Cases
        •	Create Non-functional test ideas (performance, security, usability)

    3. Analyze Risks
        •	Identify potential failure points.
        •	Highlight untestable or underspecified areas.
        •	Suggest improvements to requirement clarity or design.

    ⸻

    Output Format

    Use structured sections such as:
        1. Understanding of Requirements
        2. Test Scenarios
        3. Create Detailed Test Cases using `create_file` tool.
        4. Execute these test case and validate the software developed is as per requirement using `execute_shell_code` tool.
        5. Risks & Observations
        6. Note the test results of the `test-cases` executed as SUCCESS/FAILURE
        7. The test cases executed and their Status as SUCCESS/FAILURE.
    """,
    output_key="test_case_result",
    tools=[software_tester_toolset],
)

start_sprint = Agent(
  name='Initiate_Sprint',
  description='A Project Manager to who starts the development cycle based on the `sprints` created previously.',
  model=model,
  # Updated instruction
  instruction=f"""You are an agent who kicks of the sprint by assigning the GOALS for a particular sprint as mentioned by the user.

  **Task:**
    1) Check the STATUS of the SPRINT by reading the file SPRINT-<N>-README.md from the folder `sprints` using the tool `read_file`, where N is the current SPRINT.
    2) IF the file is PRESENT the SPRINT is COMPLETE and the user is NOT requesting for a re-starting of the SPRINT
            a) Respond with **SPRINT IS ALREADY SUCCESSFULLY IMPLEMENT**.
       ELSE
            a) Read the requirements files from folder `sprints`. They files follow the naming convention `SPRINT-<NUMBER>.txt` where NUMBER starts from `1`. Use the `read_file` tool to read the sprint files which has the requirement outlined for the particular sprint.
            b) If the **SPRINT** is greater than 1 , read the file SPRINT-<N-1>-README.md from the folder `sprints` using the tool `read_file`, where N is the current SPRINT. 
            This contains the works that has been completed in the previous SPRINT.
            c) APPEND BOTH the `requirements` and `Previous sprint progress` as the output to get a full picture of the progress. DO NOT ADD ANYTHING ELSE.
            d) STRICTLY follow the project structure as per the previous sprint.
  """,
  tools=[start_sprint_toolset],
  output_key="product_requirement"
)

product_owner = Agent(
  name='Product_Owner',
  description='A Product Owner to critique and validate the application created is correct and as per standards.',
  model=model,
  # Updated instruction
  instruction=f"""A Project manager to make sure the application delivered is correct and with proper standards.

  **REQUIREMENT:**
    ```
    {{product_requirement}}
    ```

    **CURRENT PROJECT SNAPSHOT:**
    ```
    {{summary}}
    ```

  **Task:**
    As the Product Owner, your responsibility is to critically evaluate, validate, and guide the development of any application, feature, or artifact produced by other agents or the system.

    Your duties include:

    1. **Requirements Validation**
    - Ensure the output aligns with the product vision, user needs, and business objectives.
    - Verify that functional and non-functional requirements are fully addressed.

    2. **Standards & Quality Review**
    - Assess whether the solution meets organizational and industry standards.
    - Confirm usability, accessibility, scalability, and maintainability expectations.
    - Flag deviations and provide clear corrective guidance.

    3. **Acceptance Criteria Enforcement**
    - Evaluate outputs strictly against acceptance criteria.
    - Approve only when the solution is complete, correct, and production-ready.
    - Reject with actionable feedback when gaps or ambiguities exist.

    4. **Critical Thinking & Risk Awareness**
    - Identify missing edge cases, logical inconsistencies, or potential failure points.
    - Highlight technical or product risks proactively.

    5. **Communication & Clarification**
    - Ask clarifying questions whenever requirements are incomplete or ambiguous.
    - Provide structured, concise, and high-impact feedback to improve the solution.

    Output Format:
    - Always respond with: **(a)** Validation Summary, **(b)** Issues Found, **(c)** Recommendations/Next Steps, and **(d)** Final Decision (Approved/Rejected).

    Operate with a mindset focused on delivering maximum value to users while ensuring product correctness and strategic alignment.
  """,
  tools=[],
  output_key="review_output"
)

project_manager = Agent(
  name='Project_Manager',
  description='A Project Manager to critique and validate the application created is correct and as per standards.',
  model=model,
  # Updated instruction
  instruction=f"""A Project manager to make sure the application delivered is correct and with proper standards.

  **Critique/Suggestions:**
  {{review_output if review_output is not None else ''}}

  **Task:**
    Analyze the 'Critique/Suggestions'.
    Check the ** Final Decision ** in 'Critique/Suggestions'. If it `Approved` then:
        a) You MUST provide the summary of the `SPRINT` and Respond with `Success` Message below:
        ```json
        {{
        "STATUS": "SUCCESS",
        "reasoning": "Brief summary of the sprint.
        }}

    ELSE (the critique contains actionable feedback):
        Carefully respond with the suggestions for the 'Current Sprint' as below:
        ```json
        {{
            "STATUS": "FAILURE",
            "reasoning": "Brief summary of the sprint.
        }}
  """,
  tools=[],
  output_key="final_status"
)

development_agent = LoopAgent(
    name="Software_Development_Agent",
    description= f"""
    Role:
        You are a Software Development Agent designed to plan, implement, review, and optimize software systems end-to-end. 
        You write production-quality code, reason about architecture, and use tools to execute, debug, test, and refine software. 
        You follow engineering best practices, maintain consistency, and ensure reliability.
    """,
    sub_agents=[senior_software_engineer,software_manager],
    max_iterations = 5
)



root_agent = SequentialAgent(
    name="IterativeSoftwareDevelopmentPipeline",
    sub_agents=[
        start_sprint,            # Start Development by assigning work
        development_agent,       # Loops thought development/critique process
        software_tester,         # Perform extensive testing of the application
        product_owner,           # Validates the submission is as per the requirement and standard. Provides SIGN OFF
        project_manager          # Summarizes the work and status of the Sprint.
    ],
    description="Create an software application and refines it with critique till all the acceptance criteria are met."
)