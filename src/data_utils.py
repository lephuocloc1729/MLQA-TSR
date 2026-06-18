import argparse
import re
from pathlib import Path

from src.utils import load_config, read_json, write_jsonl


def clean_html(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"<<TABLE:(.*?)\/TABLE>>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<<IMAGE:(.*?)\/IMAGE>>", r"[IMAGE: \1]", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_law_articles(config: dict):
    law_path = config["data"]["law_path"]
    output_path = config["data"]["processed_law_path"]

    raw_data = read_json(law_path)

    articles = []

    # TODO: chỉnh phần này theo schema thật của LawDB
    for idx, item in enumerate(raw_data):
        article = {
            "id": item.get("id", f"article_{idx}"),
            "law_id": item.get("law_id", ""),
            "article_id": item.get("article_id", item.get("id", "")),
            "title": item.get("title", ""),
            "content": clean_html(item.get("content", "")),
            "raw": item,
        }
        articles.append(article)

    write_jsonl(articles, output_path)
    print(f"Saved {len(articles)} articles to {output_path}")


def preprocess():
    config = load_config()
    build_law_articles(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="preprocess")
    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
