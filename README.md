# Generative Journey
Simple text-based adventure game using generative AI storytelling and played in a terminal. More of a proof of concept for comparing different models and iterating on prompting strategies then a fully fledged game. Includes a modular AI provider interface supporting both cloud and local options. Developed in part via [Google Jules](https://jules.google.com/).

## Usage
First, install the package and activate a virtual environment for it (as documented in the setup section below). Then start a game with:
```bash
play <provider> [-m <model>]
```

For further usage information see `play --help`.

### Supported Providers and Models
#### AWS Bedrock
`play bedrock` uses `amazon.nova-pro-v1:0` as its default model. (`meta.llama3-70b-instruct-v1:0` and `mistral.mistral-large-2402-v1:0` are also confirmed to work with this provider.) Requires AWS credentials via `aws login`, standard AWS env variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`), or AWS config files (`AWS_PROFILE`).

#### OpenAI
`play openai` uses `gpt-4o` as its default model. Requires the `OPENAI_API_KEY` environment variable. Can be directed at alternate provider URLs via the optional `OPENAI_BASE_URL` environment variable.

#### Local
`play local` connects to `http://localhost:1234/v1` using the OpenAI client with placeholder values for the model and API key. This provides a convenient option for connecting to compatible local LLM runtimes (LM Studio, Ollama, vLLM, or similar). 

Example usage with LM Studio:
1. Download and install [LM Studio](https://lmstudio.ai).
2. Open LM Studio, search for and download your preferred model (e.g. Llama 3, Mistral, Qwen), modify the configuration if desired, and load it into memory.
3. Navigate to the "Local Server" tab and click "Start Server". Keep the default server settings (port 1234, no authentication, disable mcp.json).
4. Run the CLI: `play local`

## Setup
Requires Python 3.10 or higher.

### Linux / macOS (bash or zsh)
1. Clone the repository and navigate into the project directory:
   ```bash
   cd generative-journey
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. Upgrade `pip` and install the package in editable mode:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

### Windows (PowerShell)
1. Clone the repository and navigate into the project directory:
   ```powershell
   cd generative-journey
   ```
2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
3. Activate the virtual environment:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
4. Upgrade `pip` and install the package in editable mode:
   ```powershell
   python -m pip install --upgrade pip
   pip install -e .
   ```
