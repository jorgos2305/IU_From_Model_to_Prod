import logging
from pathlib import Path
from dotenv import load_dotenv

def setup_logging(level=logging.INFO, log_file:str|None=None):
    
    handlers:list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        base_path = Path(__file__).resolve().parents[1]
        path_log_file = base_path / "logs" / log_file
        Path(path_log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path_log_file))

        logging.basicConfig(
            level=level,
            format="[%(levelname)s : %(name)s] - %(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers
        )