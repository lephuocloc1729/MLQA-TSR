import torch
from PIL import Image

from src.collator import IGNORE_INDEX, SFTDataCollator, format_sft_text


class FakeProcessor:
    pad_token_id = 0

    def __call__(self, text, images=None, padding=True, return_tensors=None, **kwargs):
        del padding, kwargs
        sequences = [[ord(char) % 255 + 1 for char in item] for item in text]
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = []
        attention_mask = []
        for sequence in sequences:
            pad_length = max_length - len(sequence)
            input_ids.append(sequence + [self.pad_token_id] * pad_length)
            attention_mask.append([1] * len(sequence) + [0] * pad_length)

        output = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "image_count": len(images or []),
        }
        if return_tensors == "pt":
            return output
        return {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in output.items()
        }


def write_tiny_image(path):
    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    image.save(path)


def sft_record(path, assistant='{"answer":"B"}'):
    return {
        "id": "train_1",
        "image_id": "img_1",
        "image_path": str(path),
        "messages": [
            {"role": "user", "content": "Question and legal evidence"},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_collator_output_shapes_and_label_masking(tmp_path):
    image_path = tmp_path / "img.jpg"
    write_tiny_image(image_path)
    collator = SFTDataCollator(processor=FakeProcessor())
    record = sft_record(image_path)

    batch = collator([record])
    full_text, prompt_text = format_sft_text(record, collator.assistant_marker)
    prompt_length = len(prompt_text)
    full_length = len(full_text)

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["image_count"] == 1
    assert torch.all(batch["labels"][0, :prompt_length] == IGNORE_INDEX)
    assert torch.all(batch["labels"][0, prompt_length:full_length] != IGNORE_INDEX)


def test_collator_masks_padding_tokens(tmp_path):
    image_1 = tmp_path / "img1.jpg"
    image_2 = tmp_path / "img2.jpg"
    write_tiny_image(image_1)
    write_tiny_image(image_2)
    collator = SFTDataCollator(processor=FakeProcessor())

    batch = collator(
        [
            sft_record(image_1, assistant='{"answer":"B"}'),
            sft_record(image_2, assistant='{"answer":"B","explanation":"longer"}'),
        ]
    )

    padding_positions = batch["attention_mask"] == 0
    assert torch.any(padding_positions)
    assert torch.all(batch["labels"][padding_positions] == IGNORE_INDEX)


def test_collator_can_pass_image_paths_without_loading_images(tmp_path):
    missing_path = tmp_path / "missing.jpg"
    collator = SFTDataCollator(processor=FakeProcessor(), load_images=False)

    batch = collator([sft_record(missing_path)])

    assert batch["image_count"] == 1
