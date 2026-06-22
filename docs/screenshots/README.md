# Screenshots

Place UI/architecture screenshots here and reference them from the root
[`README.md`](../../README.md) and [`../architecture.md`](../architecture.md).

Suggested captures (tenant `demo-corp`, sample documents from
[`../../data/sample_documents/`](../../data/sample_documents/)):

1. **Grounded answer with citations** — ask *"What is the remote work policy?"* and
   capture the answer, confidence, latency, and the cited sources panel.
2. **Abstention** — ask an out-of-scope question and capture the
   *"I don't know based on the available documents."* response.
3. **Document management** — the per-tenant document list with upload/delete.
4. **Chat sessions** — the sidebar with persistent, renameable conversations.
5. **Evaluation** — output of `POST /eval/run` (citation rate, abstention
   accuracy, relevance, groundedness).

Recommended format: PNG, ~1440px wide. Name files by scenario, e.g.
`grounded-answer.png`, `abstention.png`, `documents.png`, `eval-run.png`.
