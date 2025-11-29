import logging,structlog,sys,asyncio

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)
    
def main():
    from coderama_server import run_server
    asyncio.run(run_server())