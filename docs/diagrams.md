# Architecture Diagrams

Version-controlled Mermaid diagrams (GitHub renders these natively). For raster
exports, see [`screenshots/README.md`](screenshots/README.md).

---

## System architecture

```mermaid
flowchart LR
    UI["React UI"] -- "X-Tenant-ID" --> API["FastAPI"]
    API --> Router["Intent Router<br/>(llm_only / summary / hybrid)"]
    Router --> Retr["Retrieval Service<br/>(tenant-scoped)"]
    Retr --> VS["VectorStore<br/>(pgvector / HANA)"]
    Retr --> KW["Keyword search<br/>(Postgres FTS)"]
    Retr --> RR["Cross-encoder<br/>reranker (optional)"]
    Retr --> LLM["Grounded LLM<br/>(cited context only)"]
    VS --> DB[("PostgreSQL + pgvector<br/>documents / chunks / chat_sessions")]
    KW --> DB
    LLM --> API
    API --> OTel["Observability<br/>(logs + OpenTelemetry)"]
```

---

## Ingestion flow

```mermaid
flowchart TD
    F["Upload PDF / TXT / MD<br/>(X-Tenant-ID)"] --> P["Parse text<br/>(pypdf / decode)"]
    P --> C["Sentence-aware chunking<br/>(no mid-word splits)"]
    C --> E["Embed chunks<br/>(local MiniLM / OpenAI)"]
    E --> S["Persist documents + chunks<br/>tagged with tenant_id"]
    S --> DB[("PostgreSQL + pgvector")]
```

---

## Retrieval flow

```mermaid
flowchart TD
    Q["Question"] --> RW["Contextual rewrite<br/>(follow-ups)"]
    RW --> V["Vector search<br/>(cosine, tenant-scoped)"]
    RW --> K["Keyword search<br/>(FTS, tenant-scoped)"]
    V --> FU["Score fusion<br/>0.7*vector + 0.3*keyword"]
    K --> FU
    FU --> RR{"Reranker<br/>enabled?"}
    RR -- yes --> CE["Cross-encoder<br/>rerank top-N"]
    RR -- no --> TK["Top-K"]
    CE --> TK
    TK --> G{"confidence ≥<br/>threshold?"}
    G -- yes --> ANS["Grounded answer<br/>+ citations"]
    G -- no --> UNK["Abstain:<br/>'I don't know...'"]
```

---

## Tenant isolation

```mermaid
flowchart LR
    subgraph TA["Tenant A request"]
        A1["X-Tenant-ID: tenant-a"]
    end
    subgraph TB["Tenant B request"]
        B1["X-Tenant-ID: tenant-b"]
    end
    A1 -- "WHERE tenant_id = 'tenant-a'" --> DB[("chunks / documents / chat_sessions")]
    B1 -- "WHERE tenant_id = 'tenant-b'" --> DB
    DB --> A2["Only tenant-a rows"]
    DB --> B2["Only tenant-b rows"]
    A2 -. "no cross-tenant read path" .-x B2
```

Every retrieval and read query is filtered by `tenant_id` at the SQL layer; the
`VectorStore` contract requires the same, so there is no cross-tenant read path.
