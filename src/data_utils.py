import argparse
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.utils import load_config, read_json, read_jsonl, write_jsonl


TABLE_PATTERN = re.compile(r"<<TABLE:\s*(.*?)\s*/TABLE>>", re.DOTALL)
IMAGE_PATTERN = re.compile(r"<<IMAGE:\s*(.*?)\s*/IMAGE>>", re.DOTALL)
TITLE_ARTICLE_ID_PATTERN = re.compile(r"^([A-ZĐ]\.\d+(?:\.\d+)*|\d+(?:\.\d+)*)\b")


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_html(text: str) -> str:
    text = re.sub(r"<\s*br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*(td|th|tr|p|div|li|table)\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return _normalize_text(text)


def clean_html(text: str) -> str:
    """Remove LawDB table/image markers and normalize article body text."""
    if not text:
        return ""

    text = TABLE_PATTERN.sub(" ", text)
    text = IMAGE_PATTERN.sub(" ", text)
    return _strip_html(text)


def extract_images(text: str) -> list[str]:
    """Return image file names referenced by <<IMAGE: ... /IMAGE>> markers."""
    if not text:
        return []
    return [_normalize_text(match) for match in IMAGE_PATTERN.findall(text)]


def extract_tables(text: str) -> list[str]:
    """Return normalized table contents from <<TABLE: ... /TABLE>> markers."""
    if not text:
        return []
    return [_strip_html(match) for match in TABLE_PATTERN.findall(text)]


def make_article_uid(law_id: str, article_id: str) -> str:
    return f"{law_id}#{article_id}"


def normalize_article_id(raw_article_id: str, title: str) -> str:
    """Use title prefixes to recover subsection IDs missing from LawDB `id`."""
    article_id = _normalize_text(raw_article_id)
    title = _normalize_text(title)
    match = TITLE_ARTICLE_ID_PATTERN.match(title)
    if match:
        title_article_id = match.group(1)
        if article_id and title_article_id.startswith(f"{article_id}."):
            return title_article_id
    return article_id


def iter_law_articles(raw_data: list[dict]) -> Iterable[dict]:
    """Flatten the VLSP LawDB document/article structure into article records."""
    for document in raw_data:
        law_id = _normalize_text(document.get("id", ""))
        law_title = _normalize_text(document.get("title", ""))

        if not law_id:
            raise ValueError("Law document is missing required field 'id'")
        if not law_title:
            raise ValueError(f"Law document {law_id!r} is missing required field 'title'")

        articles = document.get("articles")
        if not isinstance(articles, list):
            raise ValueError(f"Law document {law_id!r} must contain an articles list")

        for article in articles:
            title = _normalize_text(article.get("title", ""))
            article_id = normalize_article_id(article.get("id", ""), title)
            raw_text = article.get("text", "")
            content = clean_html(raw_text)

            if not article_id:
                raise ValueError(f"Article in {law_id!r} is missing required field 'id'")
            if not title:
                raise ValueError(
                    f"Article {make_article_uid(law_id, article_id)!r} is missing title"
                )

            yield {
                "uid": make_article_uid(law_id, article_id),
                "law_id": law_id,
                "law_title": law_title,
                "article_id": article_id,
                "title": title,
                "content": content or title,
                "images": extract_images(raw_text),
                "tables": extract_tables(raw_text),
            }


def build_law_article_index(articles: Iterable[dict]) -> dict[str, dict]:
    """Build an O(1) lookup index and reject duplicate article UIDs."""
    index: dict[str, dict] = {}
    for article in articles:
        uid = article.get("uid") or make_article_uid(
            article.get("law_id", ""),
            article.get("article_id", ""),
        )
        if uid in index:
            raise ValueError(f"Duplicate law article UID: {uid}")
        index[uid] = article
    return index


def load_law_articles(path: str | Path) -> list[dict]:
    return read_jsonl(str(path))


def load_split_samples(config: dict, split: str) -> list[dict]:
    split_paths = {
        "train": config["data"].get("train_split_path"),
        "val": config["data"].get("val_split_path"),
        "validation": config["data"].get("val_split_path"),
    }
    if split not in split_paths:
        raise ValueError("split must be one of: train, val, validation")

    split_path = split_paths[split]
    if not split_path:
        raise KeyError(f"Missing config data path for split {split!r}")

    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{split} split not found at {path}. "
            "Run `python -m src.data_utils --mode split` after W1-03 is merged."
        )
    return read_jsonl(str(path))


def attach_train_image_path(sample: Mapping, config: dict) -> dict:
    normalized = dict(sample)
    if normalized.get("image_path"):
        return normalized

    image_id = normalized.get("image_id")
    if not image_id:
        raise ValueError("Sample is missing image_id and cannot resolve image_path")
    normalized["image_path"] = str(
        Path(config["data"]["train_image_dir"]).joinpath(str(image_id)).with_suffix(".jpg")
    )
    return normalized


def get_law_article(
    law_id: str,
    article_id: str,
    article_index: Mapping[str, dict],
) -> dict:
    """Look up one processed article by law/article ID."""
    uid = make_article_uid(_normalize_text(law_id), _normalize_text(article_id))
    try:
        return article_index[uid]
    except KeyError as exc:
        raise KeyError(f"Unknown law article UID: {uid}") from exc


def reference_uid(reference: Mapping[str, Any]) -> str:
    law_id = _normalize_text(reference.get("law_id", ""))
    article_id = _normalize_text(reference.get("article_id", ""))
    if not law_id or not article_id:
        raise ValueError("Law reference requires non-empty law_id and article_id")
    return make_article_uid(law_id, article_id)


def resolve_law_reference(
    reference: Mapping[str, Any],
    article_index: Mapping[str, dict],
) -> dict:
    """Resolve one law reference from the processed LawDB index."""
    law_id = _normalize_text(reference.get("law_id", ""))
    article_id = _normalize_text(reference.get("article_id", ""))
    return get_law_article(law_id, article_id, article_index)


def load_processed_law_article_index(config: dict) -> dict[str, dict]:
    processed_path = Path(config["data"]["processed_law_path"])
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed LawDB not found at {processed_path}. "
            "Run `python -m src.data_utils --mode preprocess` first."
        )
    return build_law_article_index(load_law_articles(processed_path))


def build_law_articles(config: dict) -> list[dict]:
    law_path = config["data"]["law_path"]
    output_path = config["data"]["processed_law_path"]

    raw_data = read_json(law_path)
    if not isinstance(raw_data, list):
        raise ValueError("LawDB JSON must be a list of law documents")

    articles = list(iter_law_articles(raw_data))
    build_law_article_index(articles)
    write_jsonl(articles, output_path)
    print(f"Saved {len(articles)} articles to {output_path}")
    return articles


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
