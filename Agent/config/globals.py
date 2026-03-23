"""Shared configuration across all project code."""

import yaml
from pathlib import Path

# Directories resolved relative to this file's location (Agent/config/globals.py)
_CONFIG_DIR = Path(__file__).resolve().parent   # Agent/config/
_AGENT_DIR = _CONFIG_DIR.parent                 # Agent/

working_dir = _AGENT_DIR
data_root_dir = _AGENT_DIR / "data"            # Where the datasets are stored
result_base_dir = _AGENT_DIR / "out"           # Where outputs are to be saved
temp_dir = result_base_dir / "temp"            # Where caches etc. are saved

embedding_model = "Alibaba-NLP/gte-base-en-v1.5"  # used for semantic search in FEVER and Averitec knowledge bases
manipulation_detection_model = _AGENT_DIR / "third_party/TruFor/weights/trufor.pth.tar"

api_key_path = _CONFIG_DIR / "api_keys.yaml"
api_keys = yaml.safe_load(open(api_key_path))

random_seed = 42 # used for sub-sampling in partial dataset testing
google_service_account_key_path = _CONFIG_DIR / "google_service_account_key.json"
firecrawl_url = "http://firecrawl:3002"  # applies to Firecrawl running in a 'firecrawl' Docker Container

def keys_configured() -> bool:
    """Returns True iff at least one key is specified."""
    return any(api_keys.values())


def configure_keys():
    """Runs a CLI dialogue where the user can set each individual API key.
    Saves them in the YAML key file."""
    for key, value in api_keys.items():
        if not value:
            user_input = input(f"Please enter your {key} (leave empty to skip): ")
            if user_input:
                api_keys[key] = user_input
                yaml.dump(api_keys, open(api_key_path, "w"))

    if not google_service_account_key_path.exists():
        user_input = input_multiline(f"Please paste the Google Service Account key file contents"
                                     f" (hit Enter to submit/skip):")
        with open(google_service_account_key_path, "w") as f:
            f.write(user_input)
    
    print("API keys configured successfully! If you want to change them, go to config/api_keys.yaml.")


def input_multiline(prompt: str):
    print(prompt)
    contents = ""
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        contents += f"\n{line}"
    return contents


if not keys_configured():
    configure_keys()
