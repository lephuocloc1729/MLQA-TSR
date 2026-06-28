#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SOURCE = Path("docs/week1-issues.md")
DEFAULT_OUTPUT_DIR = Path("data/outputs/github_issues/week1")
ISSUE_HEADING_RE = re.compile(r"^## (W\d+-\d{2}) - (.+)$", re.MULTILINE)
STOP_HEADINGS = (
    "## Suggested Parallel Work Order",
    "## Week 1 Definition Of Done",
    "## Week 2 Definition Of Done",
    "## Week 3 Definition Of Done",
    "## Week 4 Definition Of Done",
)


@dataclass(frozen=True)
class IssueSpec:
    key: str
    title: str
    body: str
    labels: list[str]
    milestone: str | None

    @property
    def github_title(self) -> str:
        return f"[{self.key}] {self.title}"


def _trim_after_issue_sections(text: str) -> str:
    indexes = [text.find(heading) for heading in STOP_HEADINGS]
    indexes = [index for index in indexes if index != -1]
    if not indexes:
        return text
    return text[: min(indexes)].rstrip()


def _extract_prefixed_value(body: str, prefix: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(prefix)}:\s*(.+)$", re.MULTILINE)
    match = pattern.search(body)
    return match.group(1).strip() if match else None


def _clean_inline_values(raw_value: str) -> list[str]:
    return [
        item.strip().strip("`")
        for item in raw_value.split(",")
        if item.strip()
    ]


def parse_issues(source: Path) -> list[IssueSpec]:
    text = _trim_after_issue_sections(source.read_text(encoding="utf-8"))
    matches = list(ISSUE_HEADING_RE.finditer(text))
    issues: list[IssueSpec] = []

    for index, match in enumerate(matches):
        key = match.group(1)
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_body = text[start:end].strip()
        body = f"## {key} - {title}\n\n{section_body}\n"

        labels_value = _extract_prefixed_value(section_body, "Labels")
        labels = _clean_inline_values(labels_value) if labels_value else []

        milestone_value = _extract_prefixed_value(section_body, "Milestone")
        milestone = milestone_value.strip("`") if milestone_value else None

        issues.append(
            IssueSpec(
                key=key,
                title=title,
                body=body,
                labels=labels,
                milestone=milestone,
            )
        )

    if not issues:
        raise ValueError(f"No weekly issue sections found in {source}")

    return issues


def write_issue_bodies(issues: list[IssueSpec], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    body_paths: dict[str, Path] = {}

    for issue in issues:
        body_path = output_dir / f"{issue.key}.md"
        body_path.write_text(issue.body, encoding="utf-8")
        body_paths[issue.key] = body_path

    return body_paths


def build_gh_command(issue: IssueSpec, body_path: Path, repo: str | None) -> list[str]:
    command = [
        "gh",
        "issue",
        "create",
        "--title",
        issue.github_title,
        "--body-file",
        str(body_path),
    ]

    if issue.milestone:
        command.extend(["--milestone", issue.milestone])

    for label in issue.labels:
        command.extend(["--label", label])

    if repo:
        command.extend(["--repo", repo])

    return command


def print_command(command: list[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or create GitHub issues from a weekly issue pack."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Markdown source file with weekly issue sections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated issue body files.",
    )
    parser.add_argument(
        "--repo",
        help="Optional GitHub repo, for example lephuocloc1729/MLQA-TSR.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Actually create issues with gh. Without this, only print commands.",
    )
    args = parser.parse_args()

    issues = parse_issues(args.source)
    body_paths = write_issue_bodies(issues, args.output_dir)

    print(f"Generated {len(issues)} issue body files in {args.output_dir}")
    print()

    if args.create and shutil.which("gh") is None:
        raise RuntimeError(
            "GitHub CLI 'gh' is not installed or not on PATH. "
            "Install it and run 'gh auth login', or run without --create."
        )

    for issue in issues:
        command = build_gh_command(issue, body_paths[issue.key], args.repo)
        if args.create:
            print(f"Creating {issue.key}: {issue.github_title}")
            subprocess.run(command, check=True)
        else:
            print_command(command)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
