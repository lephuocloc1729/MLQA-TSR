import argparse
import copy
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.schemas import Query
from src.utils import load_config, read_json, read_jsonl, write_json, write_jsonl


TABLE_PATTERN = re.compile(r"<<TABLE:\s*(.*?)\s*/TABLE>>", re.DOTALL)
IMAGE_PATTERN = re.compile(r"<<IMAGE:\s*(.*?)\s*/IMAGE>>", re.DOTALL)
TITLE_ARTICLE_ID_PATTERN = re.compile(r"^([A-ZĐ]\.\d+(?:\.\d+)*|\d+(?:\.\d+)*)\b")
QCVN_LAW_ID = "QCVN 41:2024/BGTVT"
ROAD_TRAFFIC_LAW_ID = "36/2024/QH15"
DEFAULT_VALIDATION_RATIO = 0.2


ARTICLE_REFERENCE_ALIASES: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    (ROAD_TRAFFIC_LAW_ID, "4.0"): ((ROAD_TRAFFIC_LAW_ID, "4"),),
    (ROAD_TRAFFIC_LAW_ID, "9.0"): ((ROAD_TRAFFIC_LAW_ID, "9"),),
    (ROAD_TRAFFIC_LAW_ID, "11.0"): ((ROAD_TRAFFIC_LAW_ID, "11"),),
    (ROAD_TRAFFIC_LAW_ID, "16.0"): ((ROAD_TRAFFIC_LAW_ID, "16"),),
    (ROAD_TRAFFIC_LAW_ID, "18.0"): ((ROAD_TRAFFIC_LAW_ID, "18"),),
    (ROAD_TRAFFIC_LAW_ID, "22.0"): ((ROAD_TRAFFIC_LAW_ID, "22"),),
    (ROAD_TRAFFIC_LAW_ID, "26.0"): ((ROAD_TRAFFIC_LAW_ID, "26"),),
    (QCVN_LAW_ID, "22 B.15"): ((QCVN_LAW_ID, "22"), (QCVN_LAW_ID, "B.15")),
    (QCVN_LAW_ID, "47.7"): ((QCVN_LAW_ID, "47"),),
    (QCVN_LAW_ID, "47.15"): ((QCVN_LAW_ID, "47"),),
    (QCVN_LAW_ID, "47.22"): ((QCVN_LAW_ID, "47"),),
    (QCVN_LAW_ID, "B"): ((QCVN_LAW_ID, "M.1"),),
    (QCVN_LAW_ID, "B3"): ((QCVN_LAW_ID, "B.3"),),
    (QCVN_LAW_ID, "B6"): ((QCVN_LAW_ID, "B.6"),),
    (QCVN_LAW_ID, "B.3; 41"): ((QCVN_LAW_ID, "B.3"), (QCVN_LAW_ID, "41")),
    (QCVN_LAW_ID, "B.4_x0000_"): ((QCVN_LAW_ID, "B.4"),),
    (QCVN_LAW_ID, "D.11"): ((QCVN_LAW_ID, "D.10"),),
    (QCVN_LAW_ID, "D.1777"): ((QCVN_LAW_ID, "C.45"),),
    (QCVN_LAW_ID, "F.10; 22"): ((QCVN_LAW_ID, "F.10"), (QCVN_LAW_ID, "22")),
    (QCVN_LAW_ID, "G1.1"): ((QCVN_LAW_ID, "G.1"),),
    (QCVN_LAW_ID, "G1.2"): ((QCVN_LAW_ID, "G.1"),),
    (QCVN_LAW_ID, "I.414"): ((QCVN_LAW_ID, "E.14"),),
}


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


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_law_article_index(config: dict) -> dict[str, dict]:
    processed_path = Path(config["data"]["processed_law_path"])
    if not processed_path.exists():
        build_law_articles(config)
    return build_law_article_index(load_law_articles(processed_path))


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


def normalize_article_reference(
    reference: Mapping[str, Any],
    article_index: Mapping[str, dict],
    audit: Counter[str] | None = None,
) -> list[dict[str, str]]:
    law_id = _normalize_text(reference.get("law_id", ""))
    article_id = _normalize_text(reference.get("article_id", ""))
    targets = ARTICLE_REFERENCE_ALIASES.get((law_id, article_id), ((law_id, article_id),))

    normalized_references: list[dict[str, str]] = []
    target_uids = []
    for target_law_id, target_article_id in targets:
        uid = make_article_uid(target_law_id, target_article_id)
        if uid not in article_index:
            raise KeyError(f"Unknown normalized law article UID: {uid}")
        normalized_references.append(
            {"law_id": target_law_id, "article_id": target_article_id}
        )
        target_uids.append(uid)

    raw_uid = make_article_uid(law_id, article_id)
    if audit is not None and target_uids != [raw_uid]:
        audit[f"{raw_uid} -> {', '.join(target_uids)}"] += 1

    return normalized_references


def normalize_train_sample(
    sample: Mapping[str, Any],
    config: dict,
    article_index: Mapping[str, dict],
    answer_audit: Counter[str] | None = None,
    citation_audit: Counter[str] | None = None,
) -> dict:
    normalized = copy.deepcopy(dict(sample))
    image_id = _normalize_text(normalized.get("image_id", ""))
    normalized["image_path"] = str(
        Path(config["data"]["train_image_dir"]).joinpath(image_id).with_suffix(".jpg")
    )

    raw_answer = normalized.get("answer")
    normalized_answer = unicodedata.normalize("NFC", str(raw_answer)).strip()
    if raw_answer == 40 or raw_answer == "40":
        if answer_audit is not None:
            answer_audit["40 -> A"] += 1
    elif isinstance(raw_answer, str) and raw_answer != normalized_answer:
        if answer_audit is not None:
            answer_audit[f"{raw_answer} -> {normalized_answer}"] += 1

    normalized_references: list[dict[str, str]] = []
    seen_uids: set[str] = set()
    for reference in normalized.get("relevant_articles", []):
        for normalized_reference in normalize_article_reference(
            reference,
            article_index,
            citation_audit,
        ):
            uid = make_article_uid(
                normalized_reference["law_id"],
                normalized_reference["article_id"],
            )
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            normalized_references.append(normalized_reference)
    normalized["relevant_articles"] = normalized_references

    query = Query.model_validate(normalized)
    return query.model_dump(mode="json", exclude_none=True)


def load_validated_train_samples(
    config: dict,
    article_index: Mapping[str, dict] | None = None,
) -> tuple[list[dict], dict]:
    if article_index is None:
        article_index = load_law_article_index(config)

    raw_samples = read_json(config["data"]["train_path"])
    if not isinstance(raw_samples, list):
        raise ValueError("Training JSON must be a list of samples")

    answer_audit: Counter[str] = Counter()
    citation_audit: Counter[str] = Counter()
    samples = [
        normalize_train_sample(
            sample,
            config,
            article_index,
            answer_audit=answer_audit,
            citation_audit=citation_audit,
        )
        for sample in raw_samples
    ]

    sample_ids = [sample["id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Training samples contain duplicate IDs")

    audit = {
        "answer_normalizations": dict(sorted(answer_audit.items())),
        "citation_normalizations": dict(sorted(citation_audit.items())),
    }
    return samples, audit


def grouped_train_val_split(
    samples: list[dict],
    seed: int,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    image_to_samples: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        image_to_samples[sample["image_id"]].append(sample)

    image_ids = sorted(image_to_samples)
    random.Random(seed).shuffle(image_ids)

    target_val_count = round(len(samples) * validation_ratio)
    val_image_ids: set[str] = set()
    val_count = 0
    for image_id in image_ids:
        if val_count >= target_val_count:
            break
        val_image_ids.add(image_id)
        val_count += len(image_to_samples[image_id])

    train_samples = [
        sample for sample in samples if sample["image_id"] not in val_image_ids
    ]
    val_samples = [
        sample for sample in samples if sample["image_id"] in val_image_ids
    ]

    train_samples.sort(key=lambda sample: sample["id"])
    val_samples.sort(key=lambda sample: sample["id"])
    return train_samples, val_samples


def validate_split_integrity(
    train_samples: list[dict],
    val_samples: list[dict],
    expected_samples: list[dict],
) -> None:
    train_ids = {sample["id"] for sample in train_samples}
    val_ids = {sample["id"] for sample in val_samples}
    expected_ids = {sample["id"] for sample in expected_samples}

    if train_ids & val_ids:
        raise ValueError("Train and validation splits contain overlapping sample IDs")
    if train_ids | val_ids != expected_ids:
        raise ValueError("Train and validation splits do not cover all samples")

    train_images = {sample["image_id"] for sample in train_samples}
    val_images = {sample["image_id"] for sample in val_samples}
    if train_images & val_images:
        raise ValueError("Train and validation splits contain overlapping image IDs")


def summarize_samples(samples: list[dict]) -> dict:
    return {
        "sample_count": len(samples),
        "image_count": len({sample["image_id"] for sample in samples}),
        "question_type_distribution": dict(
            sorted(
                Counter(sample.get("question_type", "Unknown") for sample in samples).items()
            )
        ),
        "answer_distribution": dict(
            sorted(Counter(sample.get("answer", "") for sample in samples).items())
        ),
    }


def build_split_manifest(
    config: dict,
    samples: list[dict],
    train_samples: list[dict],
    val_samples: list[dict],
    audit: dict,
) -> dict:
    split_config = config.get("split", {})
    train_ids = [sample["id"] for sample in train_samples]
    val_ids = [sample["id"] for sample in val_samples]
    return {
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "seed": config["project"]["seed"],
        "validation_ratio": split_config.get(
            "validation_ratio",
            DEFAULT_VALIDATION_RATIO,
        ),
        "raw_train_path": config["data"]["train_path"],
        "processed_law_path": config["data"]["processed_law_path"],
        "train_split_path": config["data"]["train_split_path"],
        "val_split_path": config["data"]["val_split_path"],
        "raw_train_sha256": file_sha256(config["data"]["train_path"]),
        "processed_law_sha256": file_sha256(config["data"]["processed_law_path"]),
        "config_hash": canonical_json_hash(
            {
                "project": config.get("project", {}),
                "data": config.get("data", {}),
                "split": config.get("split", {}),
            }
        ),
        "split_hash": canonical_json_hash({"train_ids": train_ids, "val_ids": val_ids}),
        "total_count": len(samples),
        "total_image_count": len({sample["image_id"] for sample in samples}),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_image_count": len({sample["image_id"] for sample in train_samples}),
        "val_image_count": len({sample["image_id"] for sample in val_samples}),
        "train": summarize_samples(train_samples),
        "val": summarize_samples(val_samples),
        "audit": audit,
    }


def split_train_validation(config: dict) -> dict:
    article_index = load_law_article_index(config)
    samples, audit = load_validated_train_samples(config, article_index)
    validation_ratio = config.get("split", {}).get(
        "validation_ratio",
        DEFAULT_VALIDATION_RATIO,
    )
    train_samples, val_samples = grouped_train_val_split(
        samples,
        seed=config["project"]["seed"],
        validation_ratio=validation_ratio,
    )
    validate_split_integrity(train_samples, val_samples, samples)

    write_jsonl(train_samples, config["data"]["train_split_path"])
    write_jsonl(val_samples, config["data"]["val_split_path"])
    manifest = build_split_manifest(config, samples, train_samples, val_samples, audit)
    write_json(manifest, config["data"]["split_manifest_path"])
    print(
        "Saved "
        f"{len(train_samples)} train and {len(val_samples)} validation samples "
        f"to {config['data']['train_split_path']} and {config['data']['val_split_path']}"
    )
    return manifest


def validate_data(config: dict) -> dict:
    article_index = load_law_article_index(config)
    samples, audit = load_validated_train_samples(config, article_index)

    train_path = Path(config["data"]["train_split_path"])
    val_path = Path(config["data"]["val_split_path"])
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Run `python -m src.data_utils --mode split` first")

    train_samples = read_jsonl(str(train_path))
    val_samples = read_jsonl(str(val_path))
    for sample in [*train_samples, *val_samples]:
        Query.model_validate(sample)
        for reference in sample.get("relevant_articles", []):
            uid = make_article_uid(reference["law_id"], reference["article_id"])
            if uid not in article_index:
                raise KeyError(f"Split sample cites unknown law article UID: {uid}")

    validate_split_integrity(train_samples, val_samples, samples)
    summary = {
        "total_count": len(samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_image_count": len({sample["image_id"] for sample in train_samples}),
        "val_image_count": len({sample["image_id"] for sample in val_samples}),
        "audit": audit,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


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
    elif args.mode == "split":
        split_train_validation(load_config())
    elif args.mode == "validate":
        validate_data(load_config())
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
