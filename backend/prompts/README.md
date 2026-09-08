# Nemotron Agent Prompt Library

These eight markdown contracts are the reviewable source for the runtime
prompts. Each file has exactly four sections: mission, input contract, output
contract, and bounded creativity. `app.agents.prompt_contract` renders the
shared evidence, sealed-data, and fail-closed rules around them and records the
file name plus SHA-256 in run metadata.

The reasoning model is pinned to `nvidia.nemotron-super-3-120b`. FunctionGemma
is the separate post-training target.
