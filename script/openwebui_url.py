#!/usr/bin/env python3
"""
Script to extract Open WebUI and Ollama URLs from SLURM job logs.
Similar to jupyter_url.py but for Open WebUI services.
"""

import glob
import os
import re


def get_latest_log_file(log_pattern):
    """Get the most recent log file matching the pattern."""
    list_of_files = glob.glob(log_pattern)
    if not list_of_files:
        return None

    # Get file with the largest job number (most recent)
    latest_file = max(list_of_files, key=lambda x: int(x.split("/")[-1].split(".")[-2]))
    return latest_file


def extract_urls_from_file(filename):
    """Extract URLs from a log file."""
    urls = []
    try:
        with open(filename, "r") as f:
            content = f.read()
            # Look for HTTP URLs
            matches = re.findall(r"(https?://[^\s]+)", content)
            urls.extend(matches)
    except FileNotFoundError:
        pass
    return urls


def main():
    logs_path = "logs/"

    print("🔍 Looking for Open WebUI service URLs...")
    print("=" * 50)

    # Check for dedicated URL log files first
    webui_url_file = get_latest_log_file(f"{logs_path}openwebui-url.*.log")
    ollama_url_file = get_latest_log_file(f"{logs_path}ollama-url.*.log")

    if webui_url_file:
        with open(webui_url_file, "r") as f:
            webui_url = f.read().strip()
            print(f"📱 Open WebUI: {webui_url}")

    if ollama_url_file:
        with open(ollama_url_file, "r") as f:
            ollama_url = f.read().strip()
            print(f"🤖 Ollama API: {ollama_url}")

    # If dedicated files don't exist, check main log files
    if not webui_url_file or not ollama_url_file:
        print("\n🔍 Checking main log files...")

        # Check Open WebUI logs
        openwebui_log = get_latest_log_file(f"{logs_path}openwebui.*.out")
        if openwebui_log:
            print(f"📄 Checking: {openwebui_log}")
            urls = extract_urls_from_file(openwebui_log)
            for url in urls:
                if "openwebui" in url.lower() or ":" in url:
                    print(f"📱 Open WebUI: {url}")
                elif "ollama" in url.lower():
                    print(f"🤖 Ollama API: {url}")

    print("=" * 50)
    print("💡 Access the Open WebUI in your browser using the URL above")
    print("💡 Use the Ollama API URL for direct API access")


if __name__ == "__main__":
    main()
