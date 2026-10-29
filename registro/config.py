"""Lee config.yaml y .env. Ningún otro módulo tiene rutas, modelos ni umbrales escritos a mano."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def cfg() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv(RAIZ / ".env")
    except ImportError:  # python-dotenv es opcional: también sirven variables de entorno normales
        pass
    with open(RAIZ / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta(clave: str) -> Path:
    """Ruta absoluta de una carpeta de config.yaml → rutas."""
    return RAIZ / cfg()["rutas"][clave]
