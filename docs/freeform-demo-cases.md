# Free-Form Demo Cases

Use these prompts when presenting the product-oriented Streamlit demo. They are
for live demonstration and qualitative inspection, not benchmark scoring.

## Good Presentation Flow

1. Start in retrieval-only mode to show the uploaded image, question, and
   retrieved legal evidence.
2. Use `hybrid` retrieval when the low-cost Task 1 train-example index is
   available. This combines direct LawDB text evidence with citations copied
   from similar solved training examples.
3. If a live VLM endpoint is available, switch to live mode and show that the
   answer cites only retrieved articles.
4. If the answer is uncertain, emphasize abstention and the research
   disclaimer instead of forcing a legal conclusion.

## Suggested Questions

| Scenario | Free-form question |
| --- | --- |
| Parking or stopping sign | `Tôi có được dừng hoặc đỗ xe ở vị trí trong ảnh không? Hãy nêu căn cứ pháp lý.` |
| Time-limited prohibition | `Biển báo trong ảnh có áp dụng trong khung giờ nào không? Ngoài khung giờ đó tôi được đi không?` |
| Motorcycle restriction | `Xe máy có bị cấm đi theo hướng này không? Nếu có thì dựa trên biển báo nào?` |
| Lane guidance | `Nếu tôi muốn rẽ phải hoặc ra khỏi cao tốc thì nên đi làn nào theo biển trong ảnh?` |
| Sign identification | `Biển báo chính trong ảnh thuộc nhóm biển nào và có ý nghĩa pháp lý gì?` |
| Multiple signs | `Trong ảnh có nhiều biển báo, biển nào ảnh hưởng trực tiếp đến xe con?` |
| Unclear image | `Ảnh hơi mờ; hệ thống có đủ căn cứ để kết luận không? Nếu không, hãy nói cần thêm thông tin gì.` |
| Legal citation demo | `Hãy trả lời ngắn gọn và trích dẫn điều/ký hiệu biển báo liên quan.` |

## Expected Demo Behavior

- Retrieval-only mode must work without GPU or API credentials.
- `task1`/`hybrid` retrieval requires the low-cost train-example Qdrant index;
  if it is unavailable, the demo should show diagnostics rather than hiding the
  failure.
- Live mode should answer in Vietnamese, lead with a concise conclusion, cite
  retrieved articles, and avoid official-legal-advice wording.
- If retrieved evidence is weak or the uploaded image is ambiguous, the model
  should abstain instead of guessing.

## Evaluation Dataset

Rebuild the free-form validation set from the locked validation split with:

```bash
make freeform-val
```

The output is `data/processed/freeform_val.jsonl`. Each row keeps the original
VLSP validation image, a free-form version of the question, expected citations,
and a short target answer for later retrieval/VLM evaluation.
