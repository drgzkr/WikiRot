import os
import sys
import argparse
from urllib.parse import unquote, urlparse
from pathlib import Path
import wikipedia
from movietts_v2 import generate_video_from_text  # Make sure this is properly defined

def url_to_title(url):
    """
    Extracts and decodes the article title from a Wikipedia URL.
    """
    path = urlparse(url).path
    if not path.startswith("/wiki/"):
        raise ValueError(f"Invalid Wikipedia URL: {url}")
    title = path[len("/wiki/"):]
    return unquote(title.replace("_", " "))

def generate_videos_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    for i, url in enumerate(urls):
        try:
            title = url_to_title(url)
            print(f"\n📄 [{i+1}/{len(urls)}] Processing: {title}")
            page = wikipedia.page(title)
            summary = page.summary
            generate_video_from_text(title, summary)
        except Exception as e:
            print(f"⚠️ Skipping '{url}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate videos from a list of Wikipedia article URLs.")
    parser.add_argument("url_file", type=str, help="Path to text file with one Wikipedia URL per line.")
    args = parser.parse_args()
    generate_videos_from_file(args.url_file)
