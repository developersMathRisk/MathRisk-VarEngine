"""Punto de entrada para Vercel (Python Functions): expone la aplicacion Flask de app.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402,F401
