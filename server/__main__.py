# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Enables `python -m server` and `python -m server.app` deployment mode.
Required for openenv validate python_module check.
"""



# print("[DEBUG] __main__.py executed")

from server.app import main

# print("[DEBUG] calling main()")

main()