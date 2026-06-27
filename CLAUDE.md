# Kyros — Claude Code Guidelines

## Artifact Safety

**Never delete files produced by an LLM agent** (contract.md, review.md, blueprint.md, or any output written by the Planner/Executor/Evaluator). If an artifact needs to be cleared or replaced, move it to `artifacts/` first:

```bash
mv workspace/contract.md artifacts/contract_$(date +%Y%m%d_%H%M%S).md
```

The `artifacts/` directory is the graveyard for old LLM outputs — they may contain context, decisions, or partial work that is not recoverable once deleted.
