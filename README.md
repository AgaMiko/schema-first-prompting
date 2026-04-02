# Schema-First Prompting

Agent skill for designing LLM structured output. Schema first, prompt second.

Let the Pydantic model carry the structural contract — field names, types, nesting, constraints. The prompt carries only what the schema can't: intent, tone, and rhetorical rules. Never duplicate between them.

## Install

```bash
npx skills add AgaMiko/schema-first-prompting
```

Works with Claude Code, Cursor, GitHub Copilot, Codex, Gemini CLI, and [40+ other agents](https://github.com/vercel-labs/skills).

## What's inside

Five principles, then reference sections for when you're writing code.

**1. Don't say the same thing twice** — schema owns shape, prompt owns intent. Responsibility table, no contradictions, template variables for branches, review checklist.

**2. Put reasoning first** — field order is not cosmetic. Chain-of-thought before target data, high-level decisions before details.

**3. Don't ask for what you already know** — derive fixed values in code, use separate models instead of optional fields, conditional schemas.

**4. Design for how LLMs work** — separate LLM models from API/DB models, ask for decisions not estimates, match structure to task difficulty, scope context per step.

**5. Start with the tightest schema that works** — the prompt suggests, the schema enforces. Complexity earned by failure, not speculation.

**Schema shape** — one model per shape, discriminated unions (fixed slots, tagged unions), single-string wrapper avoidance, base class guidelines.

**Field design** — naming, descriptions, nullable vs empty, closed sets, bounded extras, entity relationships, known vs unknown structure.

**Strict mode & validation** — OpenAI and Anthropic quirks, sanitizers (pre-parse cleanup), validation feedback loops with re-asking.

**Production** — versioning, evaluation with golden sets, observation and regression detection.

## Author

[Agnieszka Mikołajczyk-Bareła](https://github.com/AgaMiko)

## License

MIT