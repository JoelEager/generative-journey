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

### Supported Providers & Default Models

| Provider | Default Model | Key / Credential Requirement |
| --- | --- | --- |
| `bedrock` | `amazon.nova-pro-v1:0` | AWS credentials via `aws login`, AWS SSO, standard AWS env variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`), or AWS config files. |
| `anthropic` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` environment variable. |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` environment variable. |

### Command Options

- `-m`, `--model`: Override the default model for the selected provider.

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

If no provider argument is supplied, `play` will exit with an error message directing you back to this documentation:
```text
Error: Missing required argument 'PROVIDER'.
Supported providers: bedrock, anthropic, openai.
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

---

## Game Architecture

*(Stub - Future Implementation)*

## Gameplay Mechanics & AI Models

*(Stub - Future Implementation)*

## Contributing

*(Stub - Future Implementation)*

## License

This project is licensed under the terms of the LICENSE file included in the repository.
