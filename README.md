# Schema-First Prompting

Agent skill for designing LLM structured output. Schema first, prompt second.

Let the Pydantic model carry the structural contract — field names, types, nesting, constraints. The prompt carries only what the schema can't: intent, tone, and rhetorical rules. Never duplicate between them.

## Install

```bash
npx skills add AgaMiko/schema-first-prompting
```

Works with Claude Code, Cursor, GitHub Copilot, Codex, Gemini CLI, and [40+ other agents](https://github.com/vercel-labs/skills).

## What's inside

**Core principles** — 7 rules that govern every modeling decision: schema is the spec, one model per shape, only model what the LLM must produce, separate layers, reasoning first, no contradictions, start minimal.

**Design for how LLMs actually work** — ask for decisions not estimates, scope context per step, order fields for generation quality, match structure to task difficulty.

**Models** — discriminated unions, field hygiene, naming, nullable vs empty, bounded extras, entity relationships, reasoning fields, strict-mode handling across providers.

**Prompts** — what stays in the prompt, what the schema already covers, template variables for branches, one source of truth.

**Validation & production** — sanitizers, re-asking on validation failure, versioning, evaluation, observation.

## Author

[Agnieszka Mikołajczyk-Bareła](https://github.com/AgaMiko)

## License

MIT