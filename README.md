<div align="center">

# QwenChat

**A fast, local terminal chat client for Apple Silicon**  
*Streaming · Multi-step tools · GitHub · Plans · Memory*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-lightgrey?style=for-the-badge)
![Backends](https://img.shields.io/badge/Backends-MLX%20%7C%20GGUF-orange?style=for-the-badge)

</div>

---

## Features

| | |
|---|---|
| ⚡ | Streaming replies with token/s |
| 🧠 | Multi-step tool calling (GitHub + local + plans) |
| 📁 | Sandboxed local file tools |
| 🐙 | GitHub integration (repos, files, issues, PRs, search) |
| 📋 | Persistent plans that survive sessions |
| 💾 | Conversation history + rolling summary |
| 🎛️ | Runtime parameters (`/temp`, `/tokens`, `/preset`, …) |
| 🔄 | Numbered model switching mid-session |
| ⌨️ | Input queue (type while the model generates) |

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
git clone https://github.com/fen0dev/qwen-chat.git
cd qwen-chat

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Copy ```.env.local``` and set your model paths:

```bash
# Example:
QWEN_4BIT_PATH=/path/to/mlx-model
LFM_Q4_PATH=/path/to/model.Q4_K_M.gguf
```

Run:

```bash
python3 -m qwenchat
# or
pip install -e .
qwenchat
```
---

### Download the model

Download the model from: https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX with the following command:

```bash
hf download orcarouter/Qwen3.8-27B-Uncensored-MLX \
  --include "4-bit/*" \
  --local-dir ./models/Qwen3.8-27B-Uncensored-MLX
```

---

### Available commands inside the chat

| Command | Description |
| --- | --- |
| /models | List models and switch by number |
| /root [path] | Show or set local file sandbox root |
| /preset coding\|review\|debug\|concise\|architect | Switch system style |
| /status | Session status |
| /compact | Summarize older turns into rolling memory |
| /export [name] | Export conversation to Markdown |
| /copy | Copy last answer to clipboard |
| retry | Regenerate the last answer |
| save / load <file> / history | Conversation persistence |
| /prune | Delete sessions older than 7 days |
| /resume | Show prior context |
| /temp /tokens /top_p /think /rounds /params | Runtime generation settings |
| clear / help / exit | Standard controls |

---

## Project Structure

```
qwenchat/
  ├── app.py
  ├── config.py
  ├── ui.py
  ├── __init__.py
  ├── __main__.py
  ├── session.py
  ├── github_client.py
  ├── input_queue.py
  ├── backends/          # base + MLX + GGUF
  ├── tools/             # registry, local, plans
  ├── memory/            # conversation, history, plans
  └── models/            # empty folder where you should place open-weight model(s)
```

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

## Environment variables (optional)

| Variable | Default | Effect |
| --- | --- | --- |
| QWENCHAT_N_CTX | 4096 | GGUF context length |
| QWENCHAT_SKIP_WARMUP | unset | Set to 1 to skip warm-up |
| QWENCHAT_HISTORY_DIR | ~/.qwenchat/history | History location |
| QWENCHAT_LOCAL_ROOT | . | Local tools sandbox root |
| QWENCHAT_MLX_MAX_TOKENS | . | Local set token for harder tasks |

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
