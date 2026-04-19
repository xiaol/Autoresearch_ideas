# Persistent Compiled Knowledge as Agent Memory: From Andrej Karpathy's LLM Wiki to Agent-Native Knowledge Infrastructure

## Abstract
Andrej Karpathy's April 2026 "LLM Wiki" gist describes a simple but important design pattern: instead of repeatedly deriving answers from raw documents at query time, an LLM incrementally compiles a persistent wiki that compounds across sources and sessions [1]. This paper develops that idea into a more formal research framing. I position the pattern against classical retrieval-augmented generation (RAG) [2], connect it to earlier visions of associative knowledge systems such as Bush's Memex [3], and relate it to modern work on test-time reasoning and self-correction, including Chain-of-Thought [4], self-consistency [5], Tree of Thoughts [6], ReAct [7], Reflexion [8], Self-Refine [9], and memory-tiered agent architectures such as MemGPT [10]. I then compare six repositories that operationalize or extend the pattern in different directions: local-first compiled wikis, retrieval substrates, hosted platforms, skill templates, split-memory systems, and session-history compilers [11]-[16]. Two lightweight empirical checks are included: an offline validation pass over `llmbase`'s compilation and search tests, and a toy retrieval run with `qmd`. The central conclusion is that the LLM wiki idea is best understood not as a grand theory of "Loopy AI," but as a concrete systems pattern for durable, agent-maintained knowledge artifacts. Its stable core has three layers: immutable sources, a maintained compiled artifact, and an operational contract for ingest, query, and lint. The major design divergence is whether systems treat the compiled wiki, retrieval, or a memory hierarchy of both as the real substrate of agent memory.

---

## 1. Introduction
Most practical LLM-document systems today still follow the RAG paradigm: raw documents are indexed, relevant chunks are retrieved at query time, and the model synthesizes an answer on demand [2]. This works well for many tasks, but it has a structural weakness. Knowledge is repeatedly rediscovered instead of explicitly accumulated. Complex cross-document synthesis is paid for again on every query, contradictions remain latent until re-encountered, and useful intermediate analyses often disappear into chat history rather than entering a durable knowledge structure.

Andrej Karpathy's "LLM Wiki" note proposes a different architecture [1]. The core move is to insert a persistent, LLM-maintained markdown wiki between raw sources and future questions. Instead of treating the corpus as a passive retrieval pool, the LLM incrementally compiles it into summaries, entities, concepts, comparisons, syntheses, and logs of change. In that framing, the system's memory is not only the model's parametric weights or a retrieval index. It is also the maintained artifact itself.

That proposal has already produced an ecosystem of implementations. Some are best understood as direct realizations of the pattern. Others provide missing substrate layers such as local retrieval, hosted document workflows, or explicit memory hierarchies. Studying these implementations gives a more grounded picture of the idea than speculative claims about "wrapper death" or fully autonomous recursive cognition.

This paper asks four practical questions:

1. How should the LLM wiki pattern be situated relative to RAG, reasoning loops, and external-memory architectures?
2. What architectural elements are shared across the first wave of implementations?
3. Where do implementations diverge in meaningful ways?
4. What do actual repositories and small runnable checks suggest about the viability and limits of the pattern?

## 2. Conceptual Frame: Retrieval, Compilation, and Externalized Memory
Karpathy's note is intentionally informal, but it implies a strong architectural distinction [1].

### 2.1. RAG as query-time reconstruction
In classical RAG, the knowledge base is primarily a retrieval substrate [2]. Documents remain mostly raw. The burden of synthesis is deferred until the moment of questioning. This makes RAG flexible and scalable, but it also means that a large fraction of intellectual work remains transient. The system may retrieve the right chunks repeatedly without ever crystallizing stable, human-readable knowledge objects.

### 2.2. LLM wiki as compiled knowledge
The wiki pattern shifts work from query time to maintenance time. A source is read once, discussed, summarized, linked, and integrated into a durable artifact. The output of that compilation then becomes the default object for later use. In this sense, the wiki functions like a compiled intermediate representation for a body of knowledge. The agent need not rediscover everything every time because previous synthesis is preserved in explicit pages.

### 2.3. A memory argument
This perspective makes the wiki more than a note collection. It becomes an externalized long-term memory system. Bush's Memex remains the closest historical analogy: an associative store in which trails between documents are as valuable as the documents themselves [3]. The new ingredient is that the maintenance labor can be delegated to an LLM, making large-scale cross-referencing and revision economically feasible.

### 2.4. Why "Loopy" is only part of the story
The popular framing of "Loopy AI" emphasizes iterative reasoning, self-correction, and re-evaluation. That framing is relevant, but incomplete. A loop can improve a local answer; it does not by itself guarantee durable accumulation. Conversely, a compiled wiki can preserve knowledge without requiring deep recursive reasoning on every task. The strongest systems combine both: local iterative inference for difficult tasks, and persistent artifact maintenance for longitudinal memory.

## 3. Related Work
The LLM wiki pattern sits at the intersection of three research threads: retrieval systems, test-time reasoning, and external memory.

### 3.1. Retrieval as augmentation
Retrieval-Augmented Generation formalized the now-standard pattern of using external documents to support knowledge-intensive NLP tasks [2]. Its success established retrieval as a practical answer to the limits of purely parametric memory. The wiki pattern keeps retrieval in the loop, but reassigns it to a supporting rather than dominant role.

### 3.2. Test-time reasoning and deliberation
Chain-of-Thought prompting showed that exposing intermediate reasoning traces can substantially improve performance [4]. Self-consistency then demonstrated that sampling multiple reasoning paths and aggregating them improves reliability [5]. Tree of Thoughts generalized this into explicit search over candidate reasoning branches [6]. ReAct added interleaved tool use and reasoning [7]. Reflexion and Self-Refine introduced critique-and-revision loops in which models use verbal feedback to improve future outputs [8], [9].

These methods are highly relevant because they show that useful computation can happen at inference time, not only during pretraining. But most of them operate over transient reasoning traces. They improve a current answer; they do not automatically build a durable memory artifact.

### 3.3. External memory hierarchies
MemGPT brings the closest conceptual parallel to the wiki pattern by explicitly modeling memory tiers and paging context in and out of a limited working window [10]. The LLM wiki can be interpreted as one concrete realization of that broader principle: stable knowledge is stored outside the immediate prompt and selectively loaded back in when needed.

### 3.4. Associative knowledge systems
Bush's Memex remains the foundational conceptual reference because it frames knowledge work as the building of associative trails rather than flat indexes [3]. The wiki pattern can be seen as a markdown-native, agent-maintained, software-era Memex, in which the human curates sources and goals while the LLM performs the maintenance work that human operators usually abandon.

## 4. The Original LLM Wiki Pattern
Karpathy's gist defines a three-layer pattern [1]:

- **Raw sources**: immutable source documents that remain the ground truth.
- **The wiki**: LLM-maintained markdown pages containing summaries, entities, concepts, comparisons, and syntheses.
- **The schema**: an operational contract, such as `CLAUDE.md` or `AGENTS.md`, that tells the agent how to ingest, query, and maintain the wiki.

The note also highlights three core operations:

- **Ingest**: read a source, summarize it, update relevant pages, and record the change.
- **Query**: answer from the compiled wiki, then optionally promote useful answers back into the wiki.
- **Lint**: detect contradictions, staleness, orphan pages, missing cross-links, and gaps worth investigating.

Two implications matter especially for implementation.

First, the wiki is intended to be a *persistent, compounding artifact*, not a temporary cache of prior answers. Second, index and log files are treated as first-class operational memory. This lets small and medium-scale systems work effectively even before specialized retrieval infrastructure is introduced.

## 5. Method
On April 19, 2026, I examined the repositories most clearly connected to Karpathy's gist and its immediate implementation ecosystem:

- `Hosuke/llmbase` [11]
- `tobi/qmd` [12]
- `lucasastorian/llmwiki` [13]
- `MehmetGoekce/llm-wiki` [14]
- `Astro-Han/karpathy-llm-wiki` [15]
- `Pratiyush/llm-wiki` [16]

The comparison used three kinds of evidence:

1. README-level claims and stated design goals
2. repository structure and visible operational modules
3. small runnable checks where feasible

I also ran two lightweight local experiments:

- `llmbase`: `python3 -m pytest -q tests/test_search_raw.py tests/test_query_prefilter.py tests/test_compile.py`
- `qmd`: a toy markdown corpus with three documents, indexed locally and queried with `qmd search "emptiness" --json -n 5`

These were not benchmark-grade evaluations. They were targeted sanity checks used to separate architectural claims from executable behavior.

## 6. Comparative Analysis of Implementations

Table 1 summarizes the comparison set at a systems-design level.

| System | Primary role | Memory model | Deployment style | Main strength | Main limitation |
| --- | --- | --- | --- | --- | --- |
| `Hosuke/llmbase` [11] | Compiled wiki system | Wiki as durable memory, with raw fallback | Local-first software system | Strongest direct realization of raw -> compiled wiki -> operations | More infrastructure and moving parts than a pure markdown skill |
| `tobi/qmd` [12] | Retrieval substrate | Search as access path to markdown corpora | Local-first tool and MCP server | Excellent local search layer for larger markdown collections | Does not itself compile raw sources into durable concept pages |
| `lucasastorian/llmwiki` [13] | Hosted full-stack platform | Wiki plus platform services | Hosted/web platform | Most productized implementation with uploads, OCR, and multi-service stack | Farther from the simplicity and local-first spirit of the original gist |
| `MehmetGoekce/llm-wiki` [14] | Memory-hierarchy workflow system | L1 always-loaded memory plus L2 wiki memory | Local-first workflow and schema system | Clear operationalization of split memory for real agent work | Extends beyond the original note, making the architecture less minimal |
| `Astro-Han/karpathy-llm-wiki` [15] | Skill/template implementation | Wiki maintained through agent instructions | Installable skill/template | Lowest-friction way to adopt the pattern | Limited on indexing, storage, evaluation, and product ergonomics |
| `Pratiyush/llm-wiki` [16] | Session-history compiler | Wiki as externalized agent work memory | Local/static-site generation | Strong domain adaptation from transcript archives to durable wiki pages | Narrower source domain than general research or personal knowledge corpora |

### 6.1. `llmbase`: the clearest compiled-wiki system
`Hosuke/llmbase` is the strongest direct implementation of the compiled-wiki idea [11]. Its central claim is explicit: the LLM should compile raw material into structured wiki concepts rather than rely entirely on vector retrieval. The repository structure reflects this.

Three design choices stand out:

- an explicit operations contract in `tools/operations.py`, shared across CLI, HTTP, and MCP surfaces
- a two-layer recall model: compiled concept search plus raw-source fallback
- linting and healing treated as integral behavior rather than optional cleanup

This is the clearest example of "wiki-as-memory" in the comparison set. It extends the original note with workers, plugin-like hooks, multilingual support, and a stable extension surface, but it preserves the same ontology: raw sources -> compiled wiki -> operational interface.

### 6.2. `qmd`: the retrieval substrate, not the compiler
`tobi/qmd` should not be classified as an LLM wiki implementation in the strict sense [12]. It does not compile raw sources into durable concept pages. Instead, it provides high-quality local retrieval over markdown collections using BM25, vector search, reranking, and an MCP interface.

It is still highly relevant because Karpathy's note explicitly identifies search as an optional later-stage requirement [1]. `qmd` fills that gap by offering:

- local indexing of markdown corpora
- context hierarchies attached to collections
- MCP-native retrieval tools
- a path from small manual indexes to larger searchable corpora

In effect, `qmd` is an enabling layer for wiki systems once index-only navigation becomes insufficient. It strengthens the pattern but does not replace the compiled artifact.

### 6.3. `lucasastorian/llmwiki`: the hosted platform interpretation
`lucasastorian/llmwiki` translates the pattern into a full web product [13]. Its stack combines Next.js, FastAPI, Supabase, S3-compatible storage, OCR, upload flows, and an MCP server.

The key architectural move is to preserve the source/wiki/tools split while relocating it into a multi-service platform. This makes it the most productized implementation in the set, but it also changes the operating assumptions:

- the wiki is no longer merely local markdown in a folder
- auth and infrastructure become first-class concerns
- ingestion includes heavy document processing rather than simple markdown maintenance

This demonstrates that the pattern can scale into hosted systems, while also moving away from the local-first simplicity of the original note.

### 6.4. `MehmetGoekce/llm-wiki`: explicit L1/L2 memory hierarchy
`MehmetGoekce/llm-wiki` introduces a two-level memory model absent from Karpathy's note [14]. The repository separates:

- **L1**: always-loaded session memory for rules, credentials, preferences, and safety-critical context
- **L2**: the on-demand wiki for deeper project, workflow, and research knowledge

This is a meaningful extension. It acknowledges that some knowledge must be available before retrieval can occur. The tradeoff is that the design moves from "persistent wiki" toward "persistent wiki plus session guardrails." That makes it less pure than the original pattern, but plausibly more useful in real agent practice.

### 6.5. `Astro-Han/karpathy-llm-wiki`: the skill-template interpretation
`Astro-Han/karpathy-llm-wiki` packages the pattern as an installable agent skill rather than a software platform [15]. It includes templates, examples, and an operational specification for ingest, query, and lint.

This repository shows a low-infrastructure path for adoption: encode the pattern as reusable instructions plus markdown scaffolding. It stays very close to the spirit of the gist. Its limitation is equally clear: a skill can standardize workflow, but it cannot by itself solve indexing, concurrency, UX, storage, or evaluation.

### 6.6. `Pratiyush/llm-wiki`: session archive -> wiki -> static site
`Pratiyush/llm-wiki` applies the pattern to agent session history rather than external research documents [16]. It treats session transcripts as raw material and builds:

- a wiki layer with sources, entities, concepts, and syntheses
- a generated static site
- AI-consumable exports such as `llms.txt`, JSON-LD graphs, and machine-readable page artifacts

This is an important domain adaptation. The raw/wiki split is preserved, but the source domain becomes "agent work history" rather than papers or notes. The resulting system is closer to self-observability and externalized agent memory than to conventional personal knowledge management.

## 7. Empirical Observations

### 7.1. `llmbase` offline checks
The targeted `llmbase` test run passed all 18 selected tests on April 19, 2026. The tested areas were directly relevant to the pattern:

- raw-source fallback search
- prefiltering to prevent oversized selector prompts
- wiki index and backlink rebuilding
- section assembly and merge behavior

This matters because it shows the repository is not only describing the idea. It has executable behavior around the exact failure modes the pattern invites: prompt blow-up, raw-detail loss, backlink drift, and structural inconsistency.

### 7.2. `qmd` toy retrieval run
On a three-document demo corpus, `qmd` indexed the collection and returned the expected top result for the lexical query `emptiness`. The top hit was `doc1.md`, titled `Emptiness Notes`, with a snippet about Madhyamaka and Nagarjuna.

This is only a toy check, but it confirms that `qmd` is immediately useful as a local retrieval substrate for markdown corpora. It validates the practical role it can play underneath a compiled wiki workflow.

## 8. Synthesis and Discussion

### 8.1. Stable components
Across these repositories, several components recur consistently:

- immutable raw sources
- a maintained knowledge artifact
- an explicit operations layer for ingest, query, and lint
- cross-link maintenance or indexing
- a notion of accumulation rather than one-shot answering

These are the stable core of the LLM wiki pattern.

### 8.2. Major divergence axes
The implementation space splits along several clear dimensions:

- **compiled wiki vs retrieval engine**: `llmbase` compiles; `qmd` retrieves
- **local-first vs hosted**: `llmbase`, `qmd`, and skill repos are local-first; `lucasastorian/llmwiki` is platform-first
- **workflow skill vs software system**: `Astro-Han` and `MehmetGoekce` lean toward operational schema; `llmbase`, `lucasastorian`, and `Pratiyush` are executable systems
- **personal knowledge vs agent exhaust**: some systems ingest papers and articles; others ingest transcripts and session logs
- **single-store memory vs memory hierarchy**: the L1/L2 split in `MehmetGoekce/llm-wiki` extends the original pattern into an explicit cache hierarchy

### 8.3. Where the "Loopy AI" framing helps
The loop framing is useful when it refers to maintenance and revision:

- ingest is iterative rather than one-shot
- lint creates a regular self-check pass over the artifact
- answers can be promoted back into the wiki, creating compounding value
- retrieval can be used as a fallback when the compiled layer is incomplete

In that modest sense, the wiki pattern fits naturally inside a broader iterative-agent architecture.

### 8.4. Where the "Loopy AI" framing overreaches
The current repository ecosystem does **not** justify treating Karpathy's gist as a fully specified theory of recursive self-improving cognition. The evidence supports a narrower claim:

- persistent markdown-native knowledge artifacts are useful
- explicit ingest/query/lint operations are a robust way to manage them
- retrieval remains valuable, but often as a supporting layer
- memory hierarchies may be a natural extension once systems grow

That is a strong engineering result, but it is not the same thing as proving a grand "post-wrapper" theory of AI.

### 8.5. A clearer interpretation
The most defensible interpretation is that the LLM wiki pattern externalizes *knowledge accumulation* in the same way that Chain-of-Thought externalized intermediate reasoning [4]. CoT and its descendants show that giving inference more structure helps on a single task [4]-[9]. The wiki pattern shows that giving *knowledge maintenance* more structure helps across many tasks and over long horizons.

## 9. Limitations and Open Questions
Several questions remain unresolved across the ecosystem:

- How large can an index-driven wiki grow before specialized retrieval becomes mandatory?
- When should a useful query answer be promoted into a first-class page rather than left ephemeral?
- How should contradictions be represented: appended chronologically, merged editorially, or surfaced as explicit disputes?
- What is the right boundary between always-loaded session memory and query-time wiki retrieval?
- How should these systems be evaluated beyond demo quality: answer quality, maintenance burden, drift resistance, user trust, or longitudinal usefulness?

These are engineering questions, not merely philosophical ones, and the next phase of the ecosystem will likely be decided by them.

## 10. Conclusion
The first wave of repositories around Andrej Karpathy's April 2026 gist shows that "LLM Wiki" is real, but narrower and more concrete than hype suggests. It is not a universal theory of agent cognition. It is a practical systems pattern for building durable, agent-maintained knowledge artifacts.

The strongest implementations converge on the same backbone: immutable raw sources, compiled wiki pages, and explicit ingest/query/lint operations. Around that backbone, the ecosystem is exploring several branches: local retrieval engines, hosted document platforms, skill templates, session-history compilers, and multi-layer memory systems.

The most defensible conclusion is therefore not that AI has entered a new grand "Loopy" era. It is that persistent, markdown-native, agent-maintained knowledge systems have become a concrete and fast-evolving design space. Karpathy's gist named the pattern. The surrounding literature explains why it matters. The repositories are beginning to define its engineering reality.

***

## References
[1] Andrej Karpathy, "LLM Wiki" gist. <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>  
[2] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020. <https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html>  
[3] Vannevar Bush, "As We May Think," *The Atlantic Monthly*, July 1945. <https://www.w3.org/History/1945/vbush/vbush-all.shtml>  
[4] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," arXiv:2201.11903, 2022. <https://arxiv.org/abs/2201.11903>  
[5] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou, "Self-Consistency Improves Chain of Thought Reasoning in Language Models," arXiv:2203.11171, 2022. <https://arxiv.org/abs/2203.11171>  
[6] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan, "Tree of Thoughts: Deliberate Problem Solving with Large Language Models," arXiv:2305.10601, 2023. <https://arxiv.org/abs/2305.10601>  
[7] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao, "ReAct: Synergizing Reasoning and Acting in Language Models," arXiv:2210.03629, 2023. <https://arxiv.org/abs/2210.03629>  
[8] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao, "Reflexion: Language Agents with Verbal Reinforcement Learning," arXiv:2303.11366, 2023. <https://arxiv.org/abs/2303.11366>  
[9] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark, "Self-Refine: Iterative Refinement with Self-Feedback," arXiv:2303.17651, 2023. <https://arxiv.org/abs/2303.17651>  
[10] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez, "MemGPT: Towards LLMs as Operating Systems," arXiv:2310.08560, 2023. <https://arxiv.org/abs/2310.08560>  
[11] Hosuke, `llmbase`. <https://github.com/Hosuke/llmbase>  
[12] Tobi Lutke, `qmd`. <https://github.com/tobi/qmd>  
[13] Lucas Astorian, `llmwiki`. <https://github.com/lucasastorian/llmwiki>  
[14] Mehmet Goekce, `llm-wiki`. <https://github.com/MehmetGoekce/llm-wiki>  
[15] Astro Han, `karpathy-llm-wiki`. <https://github.com/Astro-Han/karpathy-llm-wiki>  
[16] Pratiyush, `llm-wiki`. <https://github.com/Pratiyush/llm-wiki>  
