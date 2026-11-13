# QwenChat

**Terminal chat for Qwen3.8-27B-Uncensored-MLX with conversation memory and GitHub integration.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-lightgrey)

---

## Features

- Clean and colorful terminal interface
- Full conversation memory
- GitHub integration (list repos, read files, list issues)
- Easy authentication with Personal Access Token
- Built specifically for `orcarouter/Qwen3.8-27B-Uncensored-MLX`
- Simple to install and run

---

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- At least 24 GB unified memory (recommended)
- The 4-bit version of the model (~15 GB)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/fen0dev/QwenChat.git
cd QwenChat

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download the model

Download the model from: https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX with the following command:

```bash
hf download orcarouter/Qwen3.8-27B-Uncensored-MLX \
  --include "4-bit/*" \
  --local-dir ./models/Qwen3.8-27B-Uncensored-MLX
```

---

## Usage

Before running the next commands, make sure that the models are installed and then open “.env.local“ and edit the path to your model if located elsewhere.

```bash
python -m qwenchat
```

Or after installing the package:

```bash
pip install -e .
qwenchat
```

### Available commands inside the chat

| Command   | Description                     |
|-----------|---------------------------------|
| `clear`   | Reset conversation memory       |
| `github`  | Connect / reconnect GitHub      |
| `help`    | Show help                       |
| `exit`    | Quit the application            |

---

## GitHub Integration

On first run the application will ask if you want to connect GitHub.

1. Go to [https://github.com/settings/tokens](https://github.com/settings/tokens)
2. Create a Personal Access Token with the `repo` scope
3. Paste it when asked

The token is stored locally at `~/.qwenchat_github_token`.

### Example

```
You ✦ List my repositories
Qwen ✦ Here are your latest repositories...

You ✦ Show me the README of my-project
Qwen ✦ (reads the file and answers)
```

---

## Configuration

You can edit the settings at the top of `qwenchat/config.py`:

```python
MODEL_PATH = "./models/Qwen3.8-27B-Uncensored-MLX/4-bit"
MAX_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
```

---

## Project Structure

```
QwenChat/
├── qwenchat/
    |–– models/  here you would place your models
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py
    |–– backends.py
│   ├── config.py
│   ├── github_client.py
│   ├── ui.py
│   └── tools.py
├── docs/
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## ATTENTION

When you send a prompt the model will reply first with a `MallockStackLogging` warning, which is completely harmless, and then give you full output:

Example: 
```
You ✦ List my repositories
Qwen ✦ Python(82222) MallocStackLogging: can't turn off malloc stack logging because it was not enabled.

Qwen ✦ ⚙ Calling GitHub tool: list_repos
```

---

## License

MIT License – feel free to use, modify and distribute.

---

## Credits

- Model: [orcarouter/Qwen3.8-27B-Uncensored-MLX](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX)
- Base model: Qwen3.8-27B by Alibaba
- Built with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
