#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Granular Co-localization Analysis Pipeline - Main Entry Point
Expression-Independent Object-Based Co-localization Analysis for Fluorescence Microscopy

Author: Advanced Microscopy Analysis Framework
Version: 2.0.0 - Refactored Architecture

This is the main entry point for the application.
Run this file to start the Colocalization Analysis GUI.

Usage:
    python main.py
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import application metadata
from src.utils.constants import APP_NAME, APP_VERSION, APP_AUTHOR


def main():
    """
    Main application entry point.
    Launches the Colocalization Analysis GUI.
    """
    print("=" * 70)
    print(f"{APP_NAME}")
    print(f"Version: {APP_VERSION}")
    print(f"Author: {APP_AUTHOR}")
    print("=" * 70)
    print("\nInitializing application...")

    try:
        # Import the main application module
        # We need to import from the ColocCounter2.0.py file
        # Since Python doesn't like dots in module names, we'll use importlib
        import importlib.util

        coloc_file = Path(__file__).parent / "ColocCounter2.0.py"
        if not coloc_file.exists():
            print(f"[ERROR] ColocCounter2.0.py not found at {coloc_file}")
            sys.exit(1)

        spec = importlib.util.spec_from_file_location("coloc_app", coloc_file)
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)

        print("[OK] Modules loaded successfully")
        print("\nLaunching GUI...")

        # Create and run the GUI application
        if hasattr(app_module, 'ColocalizationGUI'):
            import tkinter as tk
            root = tk.Tk()
            app = app_module.ColocalizationGUI(root)
            print("[OK] GUI initialized successfully")
            print("\nApplication is running. Close the window to exit.")
            root.mainloop()
        else:
            print("[ERROR] ColocalizationGUI class not found in module")
            sys.exit(1)

    except ImportError as e:
        print("\n[ERROR] Failed to import required modules")
        print(f"Details: {str(e)}")
        print("\nPlease ensure:")
        print("  1. All dependencies are installed (run: pip install -r requirements.txt)")
        print("  2. You're running from the correct directory")
        print("  3. The ColocCounter2.0.py file exists in the current directory")
        sys.exit(1)
    except Exception as e:
        print("\n[ERROR] Application failed to start")
        print(f"Details: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
