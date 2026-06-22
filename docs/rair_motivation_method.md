# RAIR: Motivation and Method Draft

## Motivation

Image retrieval is often formulated as a single-step matching problem: a user
submits a query, the system embeds the query and gallery images into a shared
representation space, and the nearest images are returned. This formulation is
effective when the user's intent is already clear and sufficiently aligned with
the retrieval model. In practical search scenarios, however, the initial query is
often incomplete, ambiguous, or underspecified. Users may know the target image
only partially, and their intent may evolve after inspecting early retrieval
results.

This mismatch creates three limitations for embedding-centric retrieval systems.
First, they typically assume a single-round query-to-retrieval process, while
real search is iterative. Second, they provide limited support for diagnosing why
retrieved images do or do not match the user's intent. Third, they place the
burden of query refinement on the user, who must manually decide which visual
details to add, remove, or clarify.

Recent large language models provide a natural interface for reasoning over
retrieval results, but naively applying an LLM to rewrite the query is not
sufficient. A rewrite-only approach does not explicitly inspect the retrieved
candidates, and an unconstrained LLM may hallucinate refinements that are not
grounded in the visual evidence. Similarly, directly concatenating all available
visual annotations into a query can introduce noise and over-constrain the
retrieval process.

We propose **Reasoning-Augmented Interactive Retrieval (RAIR)**, a framework for
candidate-grounded, suggestion-based image retrieval. Instead of treating
retrieval as a one-shot operation, RAIR turns retrieval into an interactive loop:
the system retrieves candidate images, analyzes their evidence, diagnoses
ambiguities and missing constraints, and proposes concrete query refinements that
the user can accept, reject, or modify.

The central idea is that retrieval results should not only be ranked, but also
interpreted. By reasoning over candidate captions and dialogue-derived visual
facts, RAIR can identify which visual details are common, ambiguous,
discriminative, or unsupported. This allows the system to guide the user toward
more effective query refinements while preserving the user's original intent.

## Problem Setting

Given an initial user query \(q_0\) and an image gallery \(\mathcal{G}\), the goal
is to retrieve a target image \(y^\*\). Unlike standard image retrieval, RAIR
assumes that the user may interact with the system over multiple turns. At each
turn \(t\), the system maintains a retrieval state consisting of the current
query, previous accepted or rejected refinements, and the latest retrieval
results.

The system does not have access to target-only annotations during retrieval.
Target annotations are used only for controlled evaluation or simulated user
feedback. During actual reasoning, RAIR can only use the current query and
evidence attached to retrieved candidate images.

## Dialogue-Derived Visual Facts

To enrich candidate evidence beyond captions, we derive visual facts from
Visual Dialog annotations. Each target image is associated with a caption and a
multi-round question-answer dialogue. We convert this dialogue into structured
facts:

- `positive_facts`: visual attributes or relations confirmed by the dialogue.
- `negative_facts`: attributes or objects explicitly denied.
- `uncertain_facts`: weak or uncertain observations.
- `visual_facts`: the union of caption-derived and dialogue-derived facts.

These facts are stored as gallery-side metadata. At retrieval time, when an image
is retrieved as a candidate, its caption and visual facts can be used as
candidate evidence for reasoning. Importantly, RAIR does not use the target
image's facts unless that image has already appeared in the retrieved candidate
pool.

## RAIR Pipeline

For each interaction turn, RAIR follows the pipeline below:

1. **Query rewriting.**  
   An LLM rewrites the current interaction state into a concise image-search
   query. The state includes the initial query, previous feedback, accepted or
   rejected suggestions, and the latest user message.

2. **Embedding-based retrieval.**  
   The rewritten query is encoded with a CLIP text encoder and used to retrieve
   candidate images from a FAISS index.

3. **Candidate evidence construction.**  
   For the top retrieved candidates, RAIR loads the image caption and
   dialogue-derived visual facts from the database.

4. **Query-aware Visual Fact Selection.**  
   Because candidate facts can be noisy, RAIR applies QVFS to select a compact
   subset of relevant and discriminative facts for each candidate.

5. **Candidate-grounded diagnosis.**  
   An LLM receives the current query and selected candidate evidence, then
   produces a compact diagnosis of the retrieval state. The diagnosis identifies
   the main intent, candidate commonalities, ambiguities, missing constraints,
   and unsupported details.

6. **Suggestion generation.**  
   The LLM generates concrete query-refinement suggestions. Each suggestion is a
   search phrase that can be accepted or rejected by the user, rather than a
   question that requires the user to formulate a new query from scratch.

7. **Interactive refinement.**  
   Accepted suggestions are incorporated into the next query state. Rejected
   suggestions are stored so that the system avoids reintroducing them in later
   turns.

## Query-aware Visual Fact Selection

A key challenge is that dialogue-derived visual facts are informative but noisy.
Some facts may be irrelevant to the current query, overly specific, common across
all candidates, or contradictory to the user's current intent. Feeding all facts
to the LLM can therefore degrade suggestion quality.

We introduce **Query-aware Visual Fact Selection (QVFS)**. For a query \(q\), a
candidate image \(c_i\), and its fact set \(F_i\), QVFS scores each fact
\(f \in F_i\) as:

\[
\text{score}(f \mid q, c_i) =
\alpha \cdot \text{sim}_{\text{CLIP}}(q, f)
+ \beta \cdot \text{lex}(q, f)
+ \gamma \cdot \text{disc}(f, C)
- \delta \cdot \text{contra}(f, q)
\]

where \(C\) is the current candidate set.

The four components capture complementary signals:

- **Semantic relevance** \(\text{sim}_{\text{CLIP}}(q, f)\): cosine similarity
  between CLIP text embeddings of the query and fact.
- **Lexical relevance** \(\text{lex}(q, f)\): normalized token overlap between
  the query and fact.
- **Discriminativeness** \(\text{disc}(f, C)\): how much the fact helps
  distinguish one candidate from the others.
- **Contradiction penalty** \(\text{contra}(f, q)\): a rule-based penalty for
  facts that conflict with the current query.

The top-\(m\) facts per candidate are retained and passed to the reasoning LLM.
In our current implementation, the default weights are:

\[
\alpha = 0.5,\quad
\beta = 0.3,\quad
\gamma = 0.2,\quad
\delta = 0.5
\]

These weights emphasize semantic relevance, use lexical overlap as an anchoring
signal, add a smaller discriminative bonus, and penalize contradictory facts.

## Suggestion-Based Interaction

RAIR differs from question-answering style interaction. Instead of asking the
user questions such as "Is the dog brown?", RAIR proposes concrete query
refinements such as "person lying beside a large brown dog" or "person resting
indoors next to a small dog." The user can accept, reject, or edit these
suggestions.

This design reduces cognitive load: the user does not need to inspect all
retrieval failures or manually invent the next query. The system analyzes
retrieval results and proposes plausible refinements grounded in candidate
evidence.

For controlled experiments, we simulate user feedback with a target-aware oracle.
The oracle can see target visual facts and accepts a generated suggestion only
when it is sufficiently supported by those facts. The RAIR system itself does
not receive target facts as input.

## Expected Contributions

This work aims to make the following contributions:

1. **Reasoning-Augmented Interactive Retrieval.**  
   We formulate image retrieval as a candidate-grounded, multi-turn interaction
   loop rather than a single query-to-ranking operation.

2. **Dialogue-derived visual fact grounding.**  
   We transform Visual Dialog annotations into positive, negative, and uncertain
   visual facts, and use them as candidate-side evidence for retrieval
   diagnosis.

3. **Query-aware Visual Fact Selection.**  
   We introduce QVFS, a lightweight fact-selection module that filters noisy
   visual facts using semantic relevance, lexical relevance, discriminativeness,
   and contradiction penalties.

4. **Suggestion-based query refinement.**  
   We design a retrieval interaction protocol in which an LLM proposes concrete
   query refinements grounded in retrieved candidates, allowing users to refine
   the search with lower cognitive effort.

## Current Empirical Status

Preliminary experiments show that naive concatenation of all visual facts can
hurt retrieval, confirming that fact selection is necessary. QVFS improves the
grounding quality of generated suggestions and increases the number of accepted
and rank-improving refinements in some settings. However, one-turn RAIR does not
yet consistently outperform rewrite-only retrieval, suggesting that query
composition and multi-turn refinement remain important areas for improvement.

Thus, the current claim should be framed carefully: RAIR and QVFS provide a
structured mechanism for reasoning and interaction in image retrieval, while
further experiments are needed to establish robust retrieval gains.

