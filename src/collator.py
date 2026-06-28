from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image


IGNORE_INDEX = -100


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def split_sft_messages(record: Mapping[str, Any]) -> tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT record must contain a non-empty messages list")

    assistant_index = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            assistant_index = index
            break
    if assistant_index is None:
        raise ValueError("SFT record must contain an assistant target message")
    if assistant_index == 0:
        raise ValueError("SFT record must contain a user prompt before the assistant")

    prompt_messages = messages[:assistant_index]
    assistant_message = messages[assistant_index]
    prompt_text = "\n\n".join(
        f"{str(message.get('role', 'user')).upper()}:\n"
        f"{message_content_to_text(message.get('content', ''))}"
        for message in prompt_messages
        if isinstance(message, Mapping)
    )
    assistant_text = message_content_to_text(assistant_message.get("content", ""))
    if not prompt_text.strip():
        raise ValueError("SFT record prompt text is empty")
    if not assistant_text.strip():
        raise ValueError("SFT record assistant target is empty")
    return prompt_text, assistant_text


def format_sft_text(record: Mapping[str, Any], assistant_marker: str) -> tuple[str, str]:
    prompt_text, assistant_text = split_sft_messages(record)
    prompt_with_marker = prompt_text + assistant_marker
    return prompt_with_marker + assistant_text, prompt_with_marker


def _supports_chat_template(processor: Any) -> bool:
    return callable(getattr(processor, "apply_chat_template", None))


def _image_content(record: Mapping[str, Any]) -> dict[str, str]:
    image_path = record.get("image_path")
    if image_path:
        return {"type": "image", "image": str(image_path)}
    return {"type": "image"}


def format_sft_chat_template_texts(
    record: Mapping[str, Any],
    processor: Any,
) -> tuple[str, str]:
    """Format SFT text with the VLM processor's native multimodal chat template."""

    prompt_text, assistant_text = split_sft_messages(record)
    user_message = {
        "role": "user",
        "content": [
            _image_content(record),
            {"type": "text", "text": prompt_text},
        ],
    }
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": assistant_text}],
    }

    full_text = processor.apply_chat_template(
        [user_message, assistant_message],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = processor.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(full_text, str) or not full_text.strip():
        raise ValueError("Processor chat template returned empty full text")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("Processor chat template returned empty prompt text")
    return full_text, prompt_text


def load_rgb_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"SFT image not found: {image_path}")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def ensure_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


@dataclass
class SFTDataCollator:
    """Processor-aware collator that trains only on assistant target tokens."""

    processor: Any
    assistant_marker: str = "\n\nASSISTANT:\n"
    ignore_index: int = IGNORE_INDEX
    load_images: bool = True
    processor_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def _images(self, records: list[Mapping[str, Any]]) -> list[Any] | None:
        if not self.load_images:
            return [record.get("image_path") for record in records]

        images: list[Image.Image] = []
        for record in records:
            image_path = record.get("image_path")
            if not image_path:
                raise ValueError("SFT record is missing image_path")
            images.append(load_rgb_image(str(image_path)))
        return images

    def _processor_call(self, texts: list[str], images: list[Any] | None) -> dict[str, Any]:
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
            **dict(self.processor_kwargs),
        )
        return dict(batch)

    def _attention_mask(self, batch: Mapping[str, Any], input_ids: torch.Tensor) -> torch.Tensor:
        if "attention_mask" in batch:
            return ensure_tensor(batch["attention_mask"]).to(dtype=torch.long)

        pad_token_id = getattr(self.processor, "pad_token_id", 0)
        return (input_ids != int(pad_token_id)).to(dtype=torch.long)

    def __call__(self, records: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            raise ValueError("SFTDataCollator requires at least one record")

        full_texts: list[str] = []
        prompt_texts: list[str] = []
        for record in records:
            if _supports_chat_template(self.processor):
                full_text, prompt_text = format_sft_chat_template_texts(
                    record,
                    self.processor,
                )
            else:
                full_text, prompt_text = format_sft_text(record, self.assistant_marker)
            full_texts.append(full_text)
            prompt_texts.append(prompt_text)

        images = self._images(records)
        batch = self._processor_call(full_texts, images)
        prompt_batch = self._processor_call(prompt_texts, images)

        input_ids = ensure_tensor(batch["input_ids"]).to(dtype=torch.long)
        attention_mask = self._attention_mask(batch, input_ids)
        prompt_input_ids = ensure_tensor(prompt_batch["input_ids"]).to(dtype=torch.long)
        prompt_attention_mask = self._attention_mask(prompt_batch, prompt_input_ids)

        labels = input_ids.clone()
        labels[attention_mask == 0] = self.ignore_index
        prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = self.ignore_index

        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels
        return batch
