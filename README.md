# generative-journey

Text-based adventure game with generative AI storytelling played in a terminal

## Setup & Installation

### Prerequisites

- Python 3.10 or higher

### Virtual Environment Setup

#### Linux / macOS (bash or zsh)

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

#### Windows

##### Command Prompt (cmd.exe)

1. Navigate into the project directory:
   ```cmd
   cd generative-journey
   ```

2. Create a virtual environment:
   ```cmd
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```cmd
   .venv\Scripts\activate.bat
   ```

4. Upgrade `pip` and install the package in editable mode:
   ```cmd
   python -m pip install --upgrade pip
   pip install -e .
   ```

##### PowerShell

1. Navigate into the project directory:
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
   *(Note: If execution policy prevents script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

4. Upgrade `pip` and install the package in editable mode:
   ```powershell
   python -m pip install --upgrade pip
   pip install -e .
   ```

## Usage

After installing the package in your active virtual environment, run the `play` command followed by the AI provider name:

```bash
play <provider> [-m <model>]
```

### Supported Providers

#### AWS Bedrock (`bedrock`)
- **Default Model**: `amazon.nova-pro-v1:0`
- **Suggested Alternate Models**:
  - Meta: `meta.llama3-70b-instruct-v1:0`
  - Mistral: `mistral.mistral-large-2402-v1:0`
- **Credential Requirement**: AWS credentials via `aws sso login`, standard AWS env variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`), or AWS config files (`AWS_PROFILE`).

#### Anthropic (`anthropic`)
- **Default Model**: `claude-3-5-sonnet-20241022`
- **Suggested Alternate Models**:
  - `claude-3-5-haiku-20241022`
  - `claude-3-opus-20240229`
- **Credential Requirement**: `ANTHROPIC_API_KEY` environment variable.

#### OpenAI (`openai`)
- **Default Model**: `gpt-4o`
- **Suggested Alternate Models**:
  - `gpt-4o-mini`
  - `o3-mini`
- **Credential Requirement**: `OPENAI_API_KEY` environment variable.

#### Local (`local`)
- **Default Model**: None required by default (only sent to server if provided via `-m` / `--model`).
- **Suggested Alternate Models**:
  - `llama-3-8b-instruct`
  - `mistral-7b-instruct`
- **Server / URL Requirement**: Any OpenAI-compatible local LLM server (e.g. LM Studio, Ollama, vLLM). Uses `http://localhost:1234/v1` by default or `LOCAL_AI_BASE_URL` when a custom URL is needed.

### Command Options

- `-m`, `--model`: Override the model for the selected provider.

### Examples

Run with AWS Bedrock (default model `amazon.nova-pro-v1:0`):
```bash
play bedrock
```

Run with Anthropic (default model `claude-3-5-sonnet-20241022`):
```bash
play anthropic
```

Run with OpenAI specifying a custom model override:
```bash
play openai -m gpt-4o-mini
```

Run with a local OpenAI-compatible server:
```bash
play local
```

If no provider argument is supplied, `play` will exit with an error message directing you back to this documentation:
```text
Error: Missing required argument 'PROVIDER'.
Supported providers: bedrock, anthropic, openai, local.
Please refer to README.md for details on supported providers, models, and API key configurations.
```

---

## Configuration & Environment Variables

### AWS Bedrock Authentication
Bedrock leverages standard AWS credential resolution using `boto3`. You can authenticate using:
- **AWS CLI Single Sign-On / Login**:
  ```bash
  aws sso login
  ```
- **Environment Variables**:
  ```bash
  export AWS_ACCESS_KEY_ID="your-access-key-id"
  export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
  export AWS_REGION="us-east-1"
  ```
- **AWS Credentials Profile (`~/.aws/credentials`)**:
  ```bash
  export AWS_PROFILE="your-profile-name"
  ```

### Anthropic API Key
Set the `ANTHROPIC_API_KEY` environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### OpenAI API Key
Set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Local Provider Setup (LM Studio & generic local servers)
The `local` provider works with any OpenAI-compatible server running locally on your machine.

#### Setting up with LM Studio

1. **Install LM Studio**:
   Download and install LM Studio from [https://lmstudio.ai](https://lmstudio.ai).

2. **Download & Load a Model**:
   Open LM Studio, search for and download your preferred model (e.g. Llama 3, Mistral, Qwen), and load it into memory.

3. **Start the Local Server**:
   Navigate to the **Local Server** tab (`<->` icon) in LM Studio and click **Start Server**. By default, it serves an OpenAI-compatible API at `http://localhost:1234/v1`.

4. **Run the CLI**:
   ```bash
   play local
   ```

#### Configuring Base URL & Options
If your local AI server (LM Studio, Ollama, vLLM, LocalAI, etc.) runs on a custom URL or port, set the `LOCAL_AI_BASE_URL` environment variable:
```bash
export LOCAL_AI_BASE_URL="http://localhost:1234/v1"
```
If your server requires an API key, set `LOCAL_AI_API_KEY`:
```bash
export LOCAL_AI_API_KEY="your-local-api-key"
```

---

## Game Architecture

*(Stub - Future Implementation)*

## Gameplay Mechanics & AI Models

*(Stub - Future Implementation)*

## Contributing

*(Stub - Future Implementation)*

## License

This project is licensed under the terms of the LICENSE file included in the repository.
