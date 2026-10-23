"""Runs src.main against the mock server named in TEST_DEPTH_ENDPOINT (used by test_main)."""

import os

import src.feed.websocket_client as wc

wc.DEPTH_ENDPOINT = os.environ["TEST_DEPTH_ENDPOINT"]

from src.main import main  # noqa: E402

main()
