import logging,click,uvicorn,os,sys

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL")

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

current_dir = os.path.dirname(os.path.abspath(__file__))
target_directory = os.path.join(current_dir, '..') 
target_directory = os.path.abspath(target_directory)
sys.path.append(target_directory)

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent_executor import ADKAgentExecutor

from .software_development_agent import root_agent as software_development_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=10003)
def main(host, port):
    # Agent card (metadata)
    agent_card = AgentCard(
        name='Software Developer Agent',
        description=software_development_agent.description,
        url=f'http://{host}:{port}',
        version="1.0.0",
        defaultInputModes=["text", "text/plain"],
        defaultOutputModes=["text", "text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="software_development",
                name="Develop the software project to the best of your ability.",
                description="Create the software project to the best of your ability.",
                tags=["build"],
                examples=[
                    "Start with Sprint 1",
                    "Lets proceed with Sprint 2",
                ],
            )
        ],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=ADKAgentExecutor(
            agent=software_development_agent,
        ),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    uvicorn.run(server.build(), host=host, port=port)


if __name__ == "__main__":
    main()