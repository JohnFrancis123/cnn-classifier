import logging

import typer
import uvicorn

from src import api
from src.config import load_config
from src.train import run_training

logging.basicConfig( #Use one shared CLI log format.
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
app = typer.Typer()


@app.command()
def train(config: str = "configs/default.yaml") -> None: #Run training from a YAML config.
    """Train CNN with YAML config."""
    cfg = load_config(config)
    run_training(cfg)


@app.command()
def serve(
    checkpoint: str = "checkpoints/best.pt",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Launch FastAPI inference server.""" #Load a checkpoint and start the API server.
    api.setup_model(checkpoint)
    uvicorn.run(api.app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()