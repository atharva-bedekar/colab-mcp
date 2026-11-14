# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import datetime
import logging
import sys
import tempfile

from fastmcp import FastMCP
from fastmcp.utilities import logging as fastmcp_logger

from colab_mcp.auth import AuthManager
from colab_mcp.models import AuthProvider
from colab_mcp.state import SessionManager
from colab_mcp.tools import register_tools

mcp = FastMCP(name="ColabMCP", on_duplicate_tools="replace")


def init_logger(logdir: str) -> None:
    log_filename = datetime.datetime.now().strftime(
        f"{logdir}/colab-mcp.%Y-%m-%d_%H-%M-%S.log"
    )
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_filename,
        level=logging.INFO,
    )
    fastmcp_logger.get_logger("colab-mcp").info("logging to %s" % log_filename)


def parse_args(v: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ColabMCP is an MCP server that lets you interact with Colab."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["auth"],
        default=None,
        help="Optional subcommand ('auth' runs interactive OAuth2 login).",
    )
    parser.add_argument(
        "-l",
        "--log",
        help="if set, use this directory as a location for logfiles (if unset, will log to %s/colab-mcp-logs/)"
        % tempfile.gettempdir(),
        action="store",
        default=tempfile.mkdtemp(prefix="colab-mcp-logs-"),
    )
    parser.add_argument(
        "--auth",
        choices=["oauth2", "adc"],
        default="oauth2",
        help="Authentication provider to use (default: oauth2).",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        default=False,
        help="Use remote copy-paste OAuth flow instead of a local loopback server.",
    )
    return parser.parse_args(v)


async def main_async(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    init_logger(args.log)

    if args.command == "auth":
        if args.remote:
            AuthManager().run_cli_login(remote=True)
        else:
            AuthManager().run_cli_login()
        return

    session_manager = SessionManager(default_auth_provider=AuthProvider(args.auth))
    register_tools(mcp, session_manager)
    async with session_manager:
        await mcp.run_async()


def main() -> None:
    asyncio.run(main_async())
