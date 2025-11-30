import asyncio,os,traceback ,json,logging,shutil,platform,subprocess,sys

LOGGING_LEVEL = "INFO"
SANDBOX_AVAILABLE = False
PLATFORM = None

os.environ["REQUIREMENT_GATHERING_AGENT_URL"] = "http://localhost:10001"
os.environ["PROJECT_PLANNER_AGENT_URL"] = "http://localhost:10002"
os.environ["SOFTWARE_DEVELOPMENT_AGENT_URL"] = "http://localhost:10003"
os.environ["LOGGING_LEVEL"] = LOGGING_LEVEL
os.environ["PROVIDER"] = "LITELLM"
os.environ["LITE_LLM_TOKEN"] = "xyz"
os.environ["DOCKER_IMAGE_TAG"] = "python:3.12.12"
os.environ["PLATFORM"] = ""
os.environ["SANDBOX_AVAILABLE"] = "FALSE"


match LOGGING_LEVEL.lower():
    case "info":
        logging.basicConfig(level=logging.INFO)
    case "debug":
        logging.basicConfig(level=logging.DEBUG)
    case "error":
        logging.basicConfig(level=logging.ERROR)
    case _:
        logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

try:
    result = subprocess.run(
        ["docker", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    SANDBOX_AVAILABLE = True
    os.environ["SANDBOX_AVAILABLE"] = "TRUE"
    arch = platform.machine().lower()
    if arch.startswith("arm") or arch.startswith("aarch"):
        PLATFORM = "linux/arm64"
        os.environ["PLATFORM"] = "linux/arm64"
except (subprocess.CalledProcessError, FileNotFoundError):
    logger.info(f" Docker Unavailable !!!!")


from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from collections.abc import AsyncIterator
from google.genai import types
import gradio as gr
from pprint import pformat



APP_NAME = 'coordinator_app'
USER_ID = 'default_user'
SESSION_ID = 'default_session'

SESSION_SERVICE = InMemorySessionService()

COORDINATOR_AGENT_RUNNER = None

POLICY_ENFORCER_AGENT_RUNNER = None


async def read_file(file_path):
    logger.info(f"========INSIDE read_file====={file_path}")
    """Function to read the content of the selected file."""
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {e}"
    return "No file selected or file not found."

async def get_workspace(file_explorer:gr.FileExplorer):
    logger.info(f"========INSIDE get_workspace=====")

    if os.getenv("WORKSPACE_DIR"):
        return gr.FileExplorer(
                    glob="**/*",
                    root_dir=os.getenv("WORKSPACE_DIR"),
                    height=250,
                    label="Workspace",
                    interactive=True,
                    file_count="single",
                )
    else:
        return file_explorer
    
async def get_current_directory():
    logger.info(f"========INSIDE get_current_directory=====")
    return gr.FileExplorer(
                    glob="**/*",
                    root_dir=os.getcwd(),
                    height=250,
                    label="Current Directory",
                    interactive=False,
                    file_count="single",
                )

async def init_agents():

    logger.info(f"=================INITIALIZING CO ORDINATOR AGENT==============")
    try:
        logger.info(f"=================INITIALIZING REQUIREMENT GATHERING AGENT==============")
        with open("requirement_gathering.log", "w") as log_file:
            process = subprocess.Popen(
                [sys.executable, "-m", "Requirement_Gathering"],
                stdout=log_file,
                stderr=log_file,
                close_fds=True
            )
            ret = process.poll()  # or process.wait()
            if ret is not None and ret != 0:
                logger.error(f"Subprocess Requirement_Gathering exited with code {ret}")
    except Exception as e:
        logger.error(
            f"""
                STATUS: FAILURE TO START REQUIREMENT GATHERING AGENT
                ERROR: {str(e)}
                TRACEBACK:{traceback.format_exc()}
                PYTHON_EXECUTABLE: {sys.executable}
    """.strip()
        )
    await asyncio.sleep(5)
    try:
        logger.info(f"=================INITIALIZING PROJECT PLANNING AGENT==============")
        with open("project_planning.log", "w") as log_file:
            process = subprocess.Popen(
                [sys.executable, "-m", "Project_Planning"],
                stdout=log_file,
                stderr=log_file,
                close_fds=True
            )
            ret = process.poll()  # or process.wait()
            if ret is not None and ret != 0:
                logger.error(f"Subprocess Project_Planning exited with code {ret}")
    except Exception as e:
        logger.error(
            f"""
                STATUS: FAILURE TO START PROJECT PLANNING AGENT
                ERROR: {str(e)}
                TRACEBACK:{traceback.format_exc()}
                PYTHON_EXECUTABLE: {sys.executable}
    """.strip()
        )
    await asyncio.sleep(5)
    try:
        logger.info(f"=================INITIALIZING SOFTWARE DEVELOPMENT AGENT==============")
        with open("software_development.log", "w") as log_file:
            process = subprocess.Popen(
                [sys.executable, "-m", "Develop_Software"],
                stdout=log_file,
                stderr=log_file,
                close_fds=True
            )
            ret = process.poll()  # or process.wait()
            if ret is not None and ret != 0:
                logger.error(f"Subprocess Develop_Software exited with code {ret}")
    except Exception as e:
        logger.error(
            f"""
                STATUS: FAILURE TO START SOFTWARE DEVELOPMENT AGENT
                ERROR: {str(e)}
                TRACEBACK:{traceback.format_exc()}
                PYTHON_EXECUTABLE: {sys.executable}
    """.strip()
        )
    await asyncio.sleep(5)

    logger.info(f"=================A2A Agents Intialized==============")
    await asyncio.sleep(5)
    
    from coordinator import initialized_coordinator_agent
    from Policy_Enforcer.policy_enforcement_agent import root_agent as policy_enforcement_agent
    global COORDINATOR_AGENT_RUNNER
    global POLICY_ENFORCER_AGENT_RUNNER

    coordinator_agent = await initialized_coordinator_agent()
    COORDINATOR_AGENT_RUNNER = Runner(
        agent=coordinator_agent,
        app_name=APP_NAME,
        session_service=SESSION_SERVICE,
    )

    POLICY_ENFORCER_AGENT_RUNNER = Runner(
        agent=policy_enforcement_agent,
        app_name=APP_NAME,
        session_service=SESSION_SERVICE,
    )

def zip_and_download():
    import tempfile
    logger.info("======Inside zip_and_download =============")
    FOLDER_TO_ZIP = os.getenv("WORKSPACE_DIR")  # Change to your folder name
    ZIP_NAME = "download.zip"
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, ZIP_NAME)
    logger.info(f"Worspace directory is {FOLDER_TO_ZIP}")
    logger.info(f"Zip file path is {zip_path}")
    
    if FOLDER_TO_ZIP is None or not os.path.exists(FOLDER_TO_ZIP):
        raise gr.Error("Directory does not exist!")

    # Create ZIP archive
    x = shutil.make_archive(zip_path.replace(".zip", ""), "zip", FOLDER_TO_ZIP)
    logger.info(f"Value of X is {x}")
    return zip_path  # Return path to the zip file

async def get_response_from_policy_agent(
    message: str,
    history: list[gr.ChatMessage],   
)-> AsyncIterator[gr.ChatMessage]:
    """Get response from policy agent."""
 
    policy_event_iterator: AsyncIterator[Event] = POLICY_ENFORCER_AGENT_RUNNER.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=types.Content(
            role='user', parts=[types.Part(text=message)]
        ),
    )

    async for event in policy_event_iterator:
        if event.is_final_response():
            final_response_text = ''
            if event.content and event.content.parts:
                final_response_text = ''.join(
                    [p.text for p in event.content.parts if p.text]
                )
            elif event.actions and event.actions.escalate:
                final_response_text = f"""
                {
                    "decision": "unsafe",
                    "reasoning": "Agent escalated: {event.error_message or "No specific message."}"
                }
                """
            if final_response_text:
                return final_response_text
        break


def get_policy_decision(policy_response_json:str):
    try:
        data = json.loads(policy_response_json)
        decision = ""
        reasoning = ""
        if "decision" in data:
            decision = data.get("decision")
            reasoning = data.get("reasoning")
        else:
            decision = "safe"
            reasoning = "safe"
    except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: for payload {policy_response_json}", e)
            decision = "safe"
            reasoning = policy_response_json
    return decision,reasoning

async def get_response_from_agent(
    message: str,
    history: list[gr.ChatMessage],
    provider_dropdown: str,
    model_name: str,
    api_key: str,
    workspace_directory: str,
    container_image_tag: str,
    timeout: str,
    sandbox: bool
) -> AsyncIterator[gr.ChatMessage]:
    """Get response from host agent."""
    if not provider_dropdown or provider_dropdown.strip() == "":
        yield gr.ChatMessage(role="assistant", content="❌ Please enter a Provider.")
        return
    else:
        os.environ["PROVIDER"] = provider_dropdown.strip()
    
    if not model_name or model_name.strip() == "":
        yield gr.ChatMessage(role="assistant", content="❌ Please enter a Model Name.")
        return
    else:
        os.environ["MODEL"] = model_name.strip()
    
    if not api_key or api_key.strip() == "":
        yield gr.ChatMessage(role="assistant", content="❌ Please enter an API key.")
        return
    else:
        if provider_dropdown.strip().lower() == "litellm":
            os.environ["LITE_LLM_TOKEN"] = api_key
        else:
            os.environ["GOOGLE_API_KEY"] = api_key
    
    if not workspace_directory or workspace_directory.strip() == "":
        yield gr.ChatMessage(role="assistant", content="❌ Please enter a Project Directory Name.")
        return
    else:
        if os.path.isabs(workspace_directory):
            WORKSPACE_DIR = workspace_directory
        else:
            os_name = os.name
            if os_name == "posix":
                WORKSPACE_DIR = os.path.join("/tmp", workspace_directory)
            elif os_name == "nt":
                WORKSPACE_DIR = os.path.join(os.path.expanduser("~"), workspace_directory)
            else:
                yield gr.ChatMessage(role="assistant", content="❌ Please enter a Project Directory Name.")
                return
        
        os.environ["WORKSPACE_DIR"] = WORKSPACE_DIR
        if not os.path.exists(WORKSPACE_DIR):
            try:
                os.makedirs(WORKSPACE_DIR,exist_ok=True)
            except (FileExistsError,Exception) as e:
                #print(f"Folder exsist {WORKSPACE_DIR}",file=sys.stderr)
                logger.error(f"Folder exsist {WORKSPACE_DIR}")
                yield gr.ChatMessage(role="assistant", content="❌ Please enter a Project Directory Name.")
                return

    if container_image_tag and container_image_tag.strip() != "":
        os.environ["DOCKER_IMAGE_TAG"] = container_image_tag.strip()
    else:
        os.environ["DOCKER_IMAGE_TAG"] = "python:3.12.12"

    if not timeout or timeout <= 0:
        yield gr.ChatMessage(role="assistant", content="❌ Please enter a Timeout.")
        return
    else:
        os.environ["TIMEOUT"] = str(timeout)
    
    if sandbox:
        SANDBOX_AVAILABLE = True
        os.environ["SANDBOX_AVAILABLE"] = "TRUE"
    else:
        SANDBOX_AVAILABLE = False
        os.environ["SANDBOX_AVAILABLE"] = "FALSE"

    try:
        if COORDINATOR_AGENT_RUNNER == None:
            await init_agents()
        
        policy_response_json = await get_response_from_policy_agent(message,history)
        decision,reasoning = get_policy_decision(policy_response_json)
        logger.info(f"Response from Policy Enforcer {decision} and reasoning is {reasoning}")
        if decision is not None and decision.lower() =="safe":
            event_iterator: AsyncIterator[Event] = COORDINATOR_AGENT_RUNNER.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=types.Content(
                    role='user', parts=[types.Part(text=message)]
                ),
            )
        else:
            yield gr.ChatMessage(
                role='assistant', content=reasoning
                        )
            return

        async for event in event_iterator:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.function_call:
                        formatted_call = f'```python\n{pformat(part.function_call.model_dump(exclude_none=True), indent=2, width=80)}\n```'
                        yield gr.ChatMessage(
                            role='assistant',
                            content=f'🛠️ **Tool Call: {part.function_call.name}**\n{formatted_call}',
                        )
                    elif part.function_response:
                        response_content = part.function_response.response
                        if (
                            isinstance(response_content, dict)
                            and 'response' in response_content
                        ):
                            formatted_response_data = response_content[
                                'response'
                            ]
                        else:
                            formatted_response_data = response_content
                        formatted_response = f'```json\n{pformat(formatted_response_data, indent=2, width=80)}\n```'
                        yield gr.ChatMessage(
                            role='assistant',
                            content=f'⚡ **Tool Response from {part.function_response.name}**\n{formatted_response}',
                        )
            if event.is_final_response():
                final_response_text = ''
                if event.content and event.content.parts:
                    final_response_text = ''.join(
                        [p.text for p in event.content.parts if p.text]
                    )
                elif event.actions and event.actions.escalate:
                    final_response_text = f'Agent escalated: {event.error_message or "No specific message."}'
                if final_response_text:
                    policy_response_json = await get_response_from_policy_agent(final_response_text,history)
                    decision,reasoning = get_policy_decision(policy_response_json)
                    logger.info(f"Response from Policy Enforcer {decision} and reasoning is {reasoning}")
                    if decision is not None and decision.lower() =="safe":
                        yield gr.ChatMessage(
                            role='assistant', content=final_response_text
                        )
                    else:
                        yield gr.ChatMessage(
                            role='assistant', content=reasoning
                        )
                break
    except Exception as e:
        logger.error(f'Error in get_response_from_agent (Type: {type(e)}): {e}')
        yield gr.ChatMessage(
            role='assistant',
            content='An error occurred while processing your request. Please check the server logs for details.',
        )


async def main():
    """Main gradio app."""
    print('Creating ADK session...')
    await SESSION_SERVICE.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print('ADK session created successfully.')

    with gr.Blocks(title='CODERAMA',fill_height=True,fill_width=True) as demo:
        with gr.Tab("Info"):
            gr.Markdown("""
            ## 📘 About this App
            🚀 Reimagining Enterprise Software Delivery with Autonomous AI Agents
            - Version: 0.0.1
            - Author: Cookie Monster
            - Description of the fields in the app
                - Select Provider : The LLM provider. We support Google (Gemini) and LiteLLM (for any other provider like Anthropic,OpenAI etc)
                - LLM Model : All Google/Gemini models will have the name like `gemini-2.5-flash` and LITELLM (for any other provider) will have the name like `openai/gpt-5-mini` OR `together_ai/openai/gpt-oss-120b`.
                - Provider API Key : API key from the provider
                - Directory Name : Your Working Directory. This is the folder where the autonomous agent will create the project and run all the executions. It has to be empty as the agent will delete any content prior to starting the project.
                - Timeout Duration in Seconds : This is a multi-agent workflow. This is the time that one agent will wait for a response from another agent. If the agent takes longer than the time set, it will timeout.
                - Enable Sandbox : This options chooses to run the project in a sandbox environment. If unchecked, the project will run in the HOST machine. Currently only Docker is supported for Sandbox.
                - Dockerhub Image : Container Image TAG From Dockerhub. Choose the image that you want to run the project with. This is to support containers for different programming languages/architecture like arm/amd and tools. ONLY works when `Enable Sandbox` is checked.
                - Refresh Button : This button gets the current snapshot of the `Working Directory`. This will show you the progress that is being made by the agent.
                - Download Button : This button download the `Working Directory` as zip.
                - For Best experience, use the app in "dark" mode.
                        
            """)
        with gr.Tab("App"):
            with gr.Row():
                gr.HTML("""
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="window-controls">
                            <span class="ctrl-close"></span>
                            <span class="ctrl-minimize"></span>
                            <span class="ctrl-maximize"></span>
                        </div>
                        <span style="font-family: 'Segoe UI', monospace; font-size: 20px; color: #E6E6E6; letter-spacing: 0.05em;">
                            🧑‍💻 CODERAMA
                        </span>
                        <span style="color: #00D9FF; font-size: 20px; margin-left: 8px;">Ultimate Vibe Coder</span>
                    </div>
                """)
                gr.HTML("""
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <span style="color: #6272a4; font-size: 12px;">v0.0.1</span>
                    </div>
                """)
            with gr.Row():
                with gr.Column():
                    provider_dropdown = gr.Dropdown(
                        label="Select Provider *",
                        choices=["LITELLM", "Google"],
                        value="LITELLM"
                    )

                    model_name = gr.Textbox(label="LLM Model *",
                                        type="text",
                                        placeholder="Name of LLM Model",
                                        show_label=True,
                    )

                    api_key = gr.Textbox(label="Provider API Key *",
                                        type="password",
                                        placeholder="sk-...",
                                        show_label=True)
                    
                    workspace_directory = gr.Textbox(label="Directory Name *",
                                        type="text",
                                        placeholder="My_Project",
                                        show_label=True,
                                        )
                    timeout = gr.Slider(minimum=600, maximum=1200, value=900, step=1, label="Timeout Duration in Seconds")
                    with gr.Row():
                        container_image_tag = gr.Textbox(label="Dockerhub Image",
                                            type="text",
                                            placeholder="python:3.12.12",
                                            show_label=True,
                        )
                        sandbox = gr.Checkbox(label="Enable Sandbox?", value=False,interactive=True)
                    with gr.Row():
                        log_content_display = gr.Textbox(
                            label="Log Content", 
                            lines=4, 
                            interactive=False
                        )
                with gr.Column(scale=5):
                    chat_interface = gr.ChatInterface(
                        get_response_from_agent,
                        additional_inputs=[provider_dropdown,model_name,api_key,workspace_directory,container_image_tag,timeout,sandbox]
                    )
                    
                    chat_interface.chatbot.height=500
                    chat_interface.chatbot.min_width=500
                    chat_interface.textbox.min_width=200
                    chat_interface.textbox.lines=7

                with gr.Column():
                    with gr.Row():
                        # File Explorer
                        file_explorer = gr.FileExplorer(
                            glob="**/*",
                            root_dir=os.getcwd(),
                            height=250,
                            label="Current Directory",
                            interactive=False,
                            file_count="single",
                        )
                        
                        file_content_display = gr.Textbox(
                            label="File Content", 
                            lines=10, 
                            interactive=False
                        )
                        file_explorer.change(
                            fn=read_file, 
                            inputs=file_explorer, 
                            outputs=file_content_display
                        )
                    with gr.Row():
                        refresh = gr.Button("Refresh",visible=True,size="md",min_width=20)
                        refresh.click(fn=get_workspace, inputs=[file_explorer],outputs=[file_explorer]).then(fn=get_current_directory,outputs=[file_explorer]).then(fn=get_workspace,inputs=[file_explorer],outputs=[file_explorer])
                        download_btn = gr.Button("Download",size="md",min_width=20)
                    download_output = gr.File(visible=True,height=80,min_width=50,interactive=True)
                    download_btn.click(zip_and_download, outputs=download_output)

    print('Launching Gradio interface...')
    demo.queue().launch(
        server_name='0.0.0.0',
        server_port=8083,
        theme=gr.themes.Ocean(),
    )
    print('Gradio application has been shut down.')


if __name__ == '__main__':
    asyncio.run(main())
