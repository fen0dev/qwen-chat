# GitHub Setup Guide

QwenChat can access your GitHub repositories so the model can list them, read files, and show issues.

## 1. Create a Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click **Generate new token** → **Generate new token (classic)**
3. Give it a name, for example: `QwenChat`
4. Select the scope: **`repo`** (Full control of private repositories)
5. Click **Generate token**
6. Copy the token immediately (you will not see it again)

## 2. Connect inside QwenChat

When you start the application it will ask:

```
Do you want to connect GitHub now? (y/n):
```

Type `y` and paste the token.

The token is saved locally at:

```
~/.qwenchat_github_token
```

## 3. Available tools

| Tool         | Description                          |
|--------------|--------------------------------------|
| `list_repos` | Lists your repositories              |
| `get_file`   | Reads a file from any of your repos  |
| `list_issues`| Shows open/closed issues of a repo   |

## Security note

Never commit your token to a public repository.  
The `.gitignore` already excludes the token file.
