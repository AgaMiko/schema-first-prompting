---
name: schema-first-prompting
description: >-
  Design Pydantic models and LLM prompt templates for structured extraction
  pipelines. Use when creating, editing, or reviewing Pydantic models that
  serve as LLM output schemas, or when writing prompt templates that pair
  with those models. Trigger: "pydantic model", "structured output",
  "extraction schema", "LLM output model", "schema design".
---

# Schema-First Prompting

Design patterns for LLM structured output: what belongs in the schema, what belongs in the prompt, and how to avoid duplication.

Most important rule: be brief, clean, elegant, and internally consistent.

## Core principles

1. **The schema is the spec.** The Pydantic model is the structural contract. The prompt carries intent, tone, and constraints the schema cannot encode. Never duplicate between them.
2. **One model per shape.** Optional fields that only apply to some shapes are a smell — use separate models or a discriminated union.
3. **Only model what the LLM must produce.** If a value is fixed once you know the variant or slot, derive it in code. The LLM must not be asked to output something the pipeline already knows.
4. **Separate models by layer.** LLM extraction shape, public API request/response, and persistence/DB rows have different fields and invariants. One "god model" for all layers leaks fields across boundaries and breaks when any layer changes.
5. **Reasoning comes first.** Because autoregressive models generate tokens sequentially, a chain-of-thought field must be the first field in the model — placing it after the target data defeats the purpose entirely.
6. **No contradictions.** If the model says "exactly 6", the prompt must not say "4-6". Contradictions create hidden bugs, worse than either version being slightly imperfect on its own.
7. **Start minimal.** Begin with the simplest schema that could work. Add reasoning fields, submodels, and constraints only when the output proves they are needed. Complexity should be earned by failure, not anticipated by speculation.

---

## Why Pydantic for LLM output

LLM completions are **untrusted input**. Same reasons Pydantic helps for HTTP or config: coercion to the right types, clear failures on bad data, and a **JSON Schema** that providers can attach to structured-output or tool/function parameters so the model is steered toward a shape you can parse.

Do not assume the model returns clean JSON strings. Raw text may include markdown fences or leading prose. **Extract JSON** (or use the provider's native structured output) **before** `model_validate` / `model_validate_json`. Sanitizers and validators are complementary: pre-parse cleanup vs. post-parse rules.

## Design for how LLMs actually work

A schema is an interface to a language model. Design it around what models are good and bad at, not what looks clean in an ORM.

### Ask for decisions, not estimates

LLMs are poor at producing absolute numeric values — durations in milliseconds, pixel coordinates, precise word counts. They are much better at **relative and categorical decisions**: which word does an effect start on, which of three severity levels applies, which item comes first. When you need a numeric output, reframe it as a choice (bins, ranks, named anchors) wherever possible. If you must ask for a number, keep the range small and well-defined in the field description.

### Scope the context per step

Dumping an entire manuscript into one call and asking for a complex nested output is a recipe for degraded quality in the tail. Break large pipelines into **focused steps**, each with its own schema, where the input is scoped to what that step needs. Use prompt caching to avoid re-sending shared context (style guides, instructions) on every call, but **restart the generation context** for each step so the model's attention is fresh. This is not just a cost optimization — it directly improves output quality on later fields.

### Ordering fields for generation quality

Autoregressive models commit to tokens left-to-right, top-to-bottom. This means field order in your schema is not cosmetic:

- **Reasoning / chain-of-thought fields first** — before the target data they inform.
- **High-level decisions before details** — a `tone` or `strategy` field before the `body_text` it should shape.
- **Independent fields before dependent ones** — if field B's quality depends on field A, A must come first in the schema.

### Match the model's grain

If a task is easy for the model, do not add reasoning fields, intermediate steps, or scaffolding "just in case." Extra fields cost tokens and can *reduce* quality by forcing the model to fill structure it does not need. Conversely, if a task is hard (multi-entity extraction, long-range consistency), invest in reasoning fields and break the work into steps. The right amount of structure depends on observed difficulty, not on how important the task feels.

## Models

### The type is the data

If a value is fixed once you know the variant or slot, do not put it in the model. Derive it in code (mapping, `match`, small helper). The LLM must not be asked to output something the pipeline already knows.

### One model per shape

Each distinct output shape gets its own model. Optional fields that only apply to some shapes are a smell — use separate models or a discriminated union (see below).

### Avoid models that are only one string

A `BaseModel` with a single `str` field adds nesting and schema noise without adding structure. Prefer a plain field on the parent with `Field(description=...)`.

```python
# Bad — wrapper adds nothing
class ClosingSummary(BaseModel):
    text: str = Field(description="2-3 sentences.")

class Report(BaseModel):
    closing: ClosingSummary

# Good
class Report(BaseModel):
    closing: str = Field(description="Closing summary: 2-3 sentences.")
```

Keep a dedicated model when there are **at least two** meaningful fields, or when you are grouping a stable sub-object at a known serialization boundary that will genuinely grow. "Will grow" means there is a concrete next field on the roadmap — not a hypothetical one.

### How you discriminate (pick one pattern)

**1. Fixed slots (no redundant `type` inside)**

When the JSON has named slots, the **key** tells you the shape. Do not add `kind` / `action` inside each child if it only repeats what the key already says.

```python
class OutlineSection(BaseModel):
    title: str = Field(description="Section heading.")
    bullets: list[str] = Field(default_factory=list, max_length=8)

class DocumentPlan(BaseModel):
    outline: OutlineSection
    summary: str = Field(description="Closing summary: 2-3 sentences.")
```

**2. Tagged union (one field, many possible shapes)**

When a **single** value must be one of several shapes (e.g. a list of heterogeneous steps), JSON has no slot name — you need a discriminator **for deserialization**, not for documentation fluff.

```python
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

class SearchStep(BaseModel):
    kind: Literal["search"] = "search"
    query: str

class AnswerStep(BaseModel):
    kind: Literal["answer"] = "answer"
    text: str

Step = Annotated[
    Union[SearchStep, AnswerStep],
    Field(discriminator="kind"),
]

class Plan(BaseModel):
    steps: list[Step]
```

Here `kind` is not redundant with the class name: it is the **wire format** tag the model must emit so the union parses. Do not also mirror it as a second field (`action`, `step_type`) saying the same thing.

**3. Conditional schema (branch omits a subtree)**

If a branch should not generate a subtree at all, use a **different root model** — not `risk_section: RiskAssessment | None` plus prompt prose saying "omit when…".

```python
class RiskAssessment(BaseModel):
    summary: str
    severity: Literal["low", "medium", "high"]

class PlanWithRisks(BaseModel):
    outline: OutlineSection
    summary: str = Field(description="Closing summary.")
    risk_section: RiskAssessment

class PlanWithoutRisks(BaseModel):
    outline: OutlineSection
    summary: str = Field(description="Closing summary.")
    # risk_section does not exist on this model at all
```

Select `PlanWithRisks` vs `PlanWithoutRisks` **before** the LLM call from application context.

### Only model what the LLM must produce

| Include | Exclude |
|---|---|
| Text, labels, lists the model must author | Values derived from variant/slot |
| Structure the model must choose | Defaults your code will apply |
| Fields downstream truly consumes from validated output | "Helper" fields you merge in after validation |

### Known vs. unknown structure

- **Known**: nested models, `max_length` / `max_items`, enums or `Literal` where the LLM must pick from a closed set. However, some providers' strict modes forbid validation keywords — see [Provider-specific strict mode](#provider-specific-strict-mode).
- **Unknown**: `dict[str, Any]`, loose `list[dict[str, Any]]` — use only where the content is genuinely open-ended. Similarly, `additionalProperties` must be set to `false` or strictly typed in strict mode, as empty dictionary annotations will cause immediate failure.

A known concept should not be `dict[str, Any]`.

### Field hygiene

- Mutable defaults: `default_factory=list`, never `[]`.
- `Field(description=...)` guides the model; avoid internal jargon.
- Drop fields nothing produces; drop "legacy" aliases.
- Do not create unnecessary submodels. Every nested model should earn its keep by adding real structure, clearer validation, or a stable reusable concept.

### Names should be reasonable

Choose names that are short, specific, and easy to read in both code and JSON Schema.

- Prefer names that describe the actual concept, not the implementation accident.
- Avoid vague names like `data`, `info`, `payload`, `value`, `type2`, or `misc`.
- Avoid ornamental naming: if `BannerCopy` says it, do not name a field `banner_copy_text_value`.
- Keep sibling fields parallel: if one field is `quote_text`, the related field should probably be `quote_source`, not `quoteAttributionLine`.
- Rename awkward names early. Small schema names spread everywhere: prompts, validators, logs, tests, and downstream code.

### Base classes

Extract a shared base only when **several** shared fields justify it. One duplicated field across two models is often clearer than a `_Base` with a single line.

### Closed sets and uncertainty

For **Enum** or **Literal** fields, include an explicit escape hatch (`OTHER`, `UNKNOWN`, …) when the model must be able to say "none of the above" without lying. That is part of the schema contract, not a prompt hack.

### Nullable vs empty string

Use `field: str | None = None` (or `Optional[str] = Field(default=None)`) when **missing** is different from **present but empty**. If you use `""` for both, validation cannot tell them apart. Be aware that under strict constrained decoding, omitted keys are not allowed. Strict mode APIs require all fields to be marked as required, using nullable types for optional values. Ensure your generated JSON schema represents optional fields explicitly as a union of types (e.g., `["string", "null"]`) while keeping the key itself in the required array.

### Bounded "extra" attributes

Open-ended `dict[str, str]` invites huge blobs. Prefer a **list of small objects** (or `(index, key, value)` tuples) with **`max_items`** in the model (stripped during strict-mode sanitization), plus a short description of the cap. Same idea for arbitrary key-value pairs: structure + limit, not an unbounded map.

### Entity relationships

If extraction output references other entities, model **IDs and relationship fields explicitly** (`parent_id`, `friend_ids: list[int]`), not only free-text names. Downstream code should not parse prose for links.

### Optional reasoning fields

A dedicated **reasoning** / **chain-of-thought** field on a **sub-object** can improve quality for that step. It costs tokens; use sparingly, not on every leaf. Do not duplicate the same instruction in the prompt if the field description already states how to reason. Crucially, **the reasoning field must be the very first field defined in the model** — the model will have already committed to its answer before it begins reasoning if it comes after the target field.

### Multimodal spatial coordinates

When extracting spatial data (like bounding boxes or coordinates) from images using vision models, enforce a normalized coordinate space. Your field description should explicitly mandate a consistent format like `[y_min, x_min, y_max, x_max]` and specify that coordinate values must be normalized (e.g., scaled from 0 to 1000 for every image) rather than relying on absolute pixel values, which models struggle to predict accurately.

### Result-or-error in the schema

When you want a **structured failure** (validation message, "could not extract") without throwing out of the LLM layer, a small wrapper model (`result: T | None`, `error: bool`, `message: str | None`) keeps control flow inside one response type. Use only where that pattern fits your API; it is not mandatory.

### Long rules for one field

If a single field has **long or subtle** extraction rules, put them in **`Field(description=...)`** for that field so they travel with the schema. That is still **one** place (the field), not the global prompt repeating every key.

## Prompts

### The schema is the spec

The JSON schema (from `model_json_schema()` or the provider's equivalent) is the structural contract. When the API supports **tool/function parameters** or **structured output** tied to that schema, put shape there and keep the user/system messages to **task + context + behavior** — not a prose duplicate of every field.

If native structured outputs or tools are unsupported and you must inject the schema into the prompt context, avoid injecting raw JSON Schema (`model_json_schema()`), as it is highly token-inefficient. Instead, use type-definitions (which look like TypeScript interfaces or Pydantic pseudo-code), as this lossless compression technique can reduce token usage by up to 60% while being clearer to the model's attention mechanism.

The prompt should **not** restate:

- Field names, types, nesting, required vs optional.
- Defaults already in the model.
- Lists of keys the schema already enumerates.

Keep in the prompt:

- **Intent**: tone, audience, what "good" looks like.
- **Constraints the schema cannot encode**: "at most 50 words", "no proper nouns from the input", "do not repeat the previous section."
- **Inputs**: pasted documents, `{variables}`.
- **Conditional blocks** via template variables, not by asking the model to "ignore" a section.

### Template variables for branches

```python
USER_PROMPT = """You are extracting a plan.

{extra_instructions}

Source text:
{source}
"""
# Caller sets extra_instructions="" or extra_instructions=RISKS_BLOCK when using PlanWithRisks.
```

The model never sees instructions for a branch that is not in the schema.

### One source of truth

| What | Where |
|---|---|
| Shape, types, limits | Pydantic model |
| Fixed values from variant/slot | Your code after validation |
| Rhetorical / quality rules | Prompt |
| "Which schema today?" | Caller (conditional model + prompt) |

If something appears in both the schema description and the prompt, delete the duplicate.

When you review a prompt/model pair, check three things explicitly:

- **Names match**: the prompt uses the same field and concept names as the schema.
- **Constraints match**: counts, limits, optionality, and branch behavior are identical.
- **Responsibility matches**: the prompt asks only for what the schema expects, and the schema models only what the LLM must produce.

## Provider-specific strict mode

Not all providers handle JSON Schema validation keywords the same way. Know what your target supports before relying on field-level constraints.

**OpenAI (`strict=True`):** The underlying parser explicitly forbids `maxLength`, `maxItems`, `minimum`, `maximum`, and similar validation keywords. Sending a Pydantic model with `Field(ge=0, le=150)` results in an immediate 400 error. Implement a schema sanitizer that strips these constraints from the JSON Schema before sending it to the API, while keeping the unmodified Pydantic model for post-generation Python-side validation.

**Anthropic (tool use):** Tool input schemas accept standard JSON Schema validation keywords including `maxLength`, `minLength`, `pattern`, `minimum`, `maximum`, `minItems`, and `maxItems`. No sanitization is needed for these constraints — Pydantic models with `Field(ge=0, le=150)` or `Field(max_length=500)` work as-is when passed as tool parameter schemas. However, the model may still occasionally violate soft constraints, so keep Python-side validation as a safety net.

**General strategy:** Write your Pydantic models with full validation (`max_length`, `ge`, `le`, `max_items`, etc.). Then apply a provider-specific sanitizer only where required. This gives you one authoritative model with the tightest constraints, and a thin adapter layer per provider.

## Sanitizers (pre-validation)

Run on **parsed** JSON (dict) from the model **before** `model_validate()`, once fences/prose are stripped if you are not using native structured output.

**Do:** coerce `None` → `""`, list → joined string where needed, strip overlong strings, `pop()` keys that are not on the model (LLM hallucinated extras).

**Don't:** re-implement defaults or `Literal` enforcement the validator already applies; keep dead branches for old shapes.

## Validation feedback loop (re-asking)

When `model_validate()` fails due to hallucinations or missed constraints, do not simply drop the data. Catch the Pydantic `ValidationError` and feed the exact error message (e.g., `"Value error, Name must contain a space"`) back to the LLM in a new user prompt, commanding it to self-correct its previous output. Libraries like Instructor automate this retry loop by catching validation errors and returning them directly to the model alongside the original completion payload.

## Prompts in production

Prompts are **artifacts**, not immortal strings in code.

- **Templates**: Separate fixed wording from runtime data (variables, branches). Data and structure are not one concatenated blob.
- **Versioning**: Track changes in source control. When behavior shifts, you need a diff and a rollback story.
- **Evaluation**: Keep a **small golden set** of inputs with expected or acceptable outputs; rerun when the model or prompt changes. Subjective tasks still need criteria (length, must-include fields, forbidden patterns).
- **Observation**: Log latency, token use, and validation failures per prompt/version so regressions surface before users report them.

## General hygiene

- Prefer modern builtins: `dict[str, Any]`, `list[str]`.
- Skip `__all__` unless the package is a published API.
- No unused imports, dead constants, or commented-out blocks.
- Elegant code is usually the simplest code that keeps the contract precise. Prefer clarity over cleverness, and compactness over repetition.