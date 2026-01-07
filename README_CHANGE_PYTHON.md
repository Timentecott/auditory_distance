This workspace sets the VS Code Python interpreter to the user's local Python installation.

If you need to change the interpreter:

1. Open Command Palette (Ctrl+Shift+P) -> "Python: Select Interpreter" and pick the desired interpreter.
2. Or edit `.vscode/settings.json` and adjust `python.defaultInterpreterPath` to the full path of the Python executable.

Recommended workflow:

- Create a venv with the chosen interpreter:
  `C:\\path\\to\\python.exe -m venv .venv`
- Activate and install requirements:
  `.venv\\Scripts\\activate` (CMD) or `.\.venv\\Scripts\\Activate.ps1` (PowerShell)
  `pip install -r requirements.txt`

If VS Code still uses the Microsoft Store shim, disable the App execution aliases in Windows Settings -> Apps -> Advanced app settings -> App execution aliases.
