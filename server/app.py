# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the DataGym Environment.

Supported deployment modes:
    docker        — Dockerfile CMD: uvicorn server.app:app
    openenv_serve — openenv serve (uses entrypoint: server.app:main)
    uv_run        — uv run server  (pyproject.toml scripts entry)
    python_module — python -m server
"""

import os
import sys




# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Compute paths from __file__ so this works regardless of CWD.
# Required because openenv validate, uv run, and python -m each set a
# different working directory when they import this module.
_SERVER_DIR   = os.path.dirname(os.path.abspath(__file__))  # .../server
_PROJECT_ROOT = os.path.dirname(_SERVER_DIR)                 # project root

for _p in (_PROJECT_ROOT, _SERVER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports ────────────────────────────────────────────────────────────────────
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e



from models import DatagymAction, DatagymObservation  # always found via _PROJECT_ROOT


from server.DataGym_environment import DatagymEnvironment


# ── App ────────────────────────────────────────────────────────────────────────
app = create_app(
    DatagymEnvironment,
    DatagymAction,
    DatagymObservation,
    env_name="DataGym",
    max_concurrent_envs=10,
)


# ── Entry point ────────────────────────────────────────────────────────────────
def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    
    """
    Start the DataGym environment server.

    Passing the `app` object (not a string) so uvicorn never needs to
    re-import the module — works correctly regardless of CWD.
    """

    # print("[DEBUG] main() CALLED")
    # print("[DEBUG] host:", host, "port:", port)
    import uvicorn
    uvicorn.run(
        "server.app:app",   # ✅ important
        host=host,
        port=port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DataGym environment server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)