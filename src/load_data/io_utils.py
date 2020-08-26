from dotenv import load_dotenv, find_dotenv
import os

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

data_path = os.environ.get("DATA_PATH")
raw_path = os.path.join(data_path, "raw", "")
interim_path = os.path.join(data_path, "interim", "")
processed_path = os.path.join(data_path, "processed", "")