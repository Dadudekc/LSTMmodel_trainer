# Packaging Instructions

These steps outline how to create a standalone executable using PyInstaller.

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```
2. Run PyInstaller on the main entry point:
   ```bash
   pyinstaller --onefile src/main.py
   ```
   This will generate a binary in the `dist/` directory.
3. Test the produced executable on your platform to ensure it launches correctly.

Additional configuration such as including data files or setting icons can be added to a `.spec` file. Refer to the PyInstaller documentation for platform‑specific options.
