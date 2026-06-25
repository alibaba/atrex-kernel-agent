# OpenCode Unified Model Configuration Guide

When you have only one model provider (such as an internal proxy for Anthropic) and need all agents and sub-agents to use the same model, follow this guide to configure.

## Problem

OpenCode's oh-my-opencode plugin includes multiple built-in agents (oracle, explore, librarian, etc.) and multiple task categories (deep, quick, ultrabrain, etc.), each with **its own independent default model configuration**. If not explicitly overridden, they will attempt to call models that your provider does not support, causing sub-agent startup failures.

## Involved Configuration Files

```
~/.config/opencode/
├── opencode.json          # ① Provider and model definitions
└── oh-my-opencode.json    # ② Agent/Category model mapping (critical!)
```

## Configuration Steps

### Step 1: `opencode.json` — Define Provider

Register your API endpoint and available models in `provider`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "my-provider": {                          // ← Provider name, custom
      "npm": "@ai-sdk/anthropic",             // ← SDK type
      "name": "my-provider",
      "options": {
        "baseURL": "https://your-api-proxy/v1",
        "apiKey": "your-api-key"
      },
      "models": {
        "claude-opus-4-6": {                  // ← Model ID
          "name": "claude-opus-4-6",          // ← Actual upstream model name
          "limit": {
            "context": 400000,
            "output": 65536
          }
        }
      }
    }
  },
  "plugin": ["oh-my-opencode@latest"]
}
```

This step defines **the only available model** `my-provider/claude-opus-4-6`.

### Step 2: `oh-my-opencode.json` — Unify All Agents and Categories

This is the **core** step. You must explicitly point each agent and category to your model, using the format `"providerName/modelID"`:

```jsonc
{
  "$schema": "https://raw.githubusercontent.com/code-yeongyu/oh-my-opencode/dev/assets/oh-my-opencode.schema.json",

  // ── All built-in Agents ──
  "agents": {
    "hephaestus":       { "model": "my-provider/claude-opus-4-6" },
    "oracle":           { "model": "my-provider/claude-opus-4-6" },
    "librarian":        { "model": "my-provider/claude-opus-4-6" },
    "explore":          { "model": "my-provider/claude-opus-4-6" },
    "multimodal-looker":{ "model": "my-provider/claude-opus-4-6" },
    "prometheus":       { "model": "my-provider/claude-opus-4-6" },
    "metis":            { "model": "my-provider/claude-opus-4-6" },
    "momus":            { "model": "my-provider/claude-opus-4-6" },
    "atlas":            { "model": "my-provider/claude-opus-4-6" }
  },

  // ── All Task Categories (determines the model used by sub-agent when task() is called) ──
  "categories": {
    "visual-engineering": { "model": "my-provider/claude-opus-4-6" },
    "ultrabrain":         { "model": "my-provider/claude-opus-4-6" },
    "deep":               { "model": "my-provider/claude-opus-4-6" },
    "artistry":           { "model": "my-provider/claude-opus-4-6" },
    "quick":              { "model": "my-provider/claude-opus-4-6" },
    "unspecified-low":    { "model": "my-provider/claude-opus-4-6" },
    "unspecified-high":   { "model": "my-provider/claude-opus-4-6" },
    "writing":            { "model": "my-provider/claude-opus-4-6" }
  }
}
```

### Complete List of Items to Override

**Agents** (invoked via `subagent_type=`):

| Agent Name | Purpose |
|------------|---------|
| `hephaestus` | GPT model-specific build agent |
| `oracle` | Read-only high-quality reasoning advisor |
| `librarian` | External documentation/OSS search |
| `explore` | Codebase structure exploration |
| `multimodal-looker` | Image/PDF analysis |
| `prometheus` | Work plan generation |
| `metis` | Pre-planning analysis |
| `momus` | Plan review |
| `atlas` | Knowledge base management |**Categories** (determines the model used by Sisyphus-Junior, called via `category=`):

| Category | Purpose |
|----------|------|
| `visual-engineering` | Frontend/UI/UX |
| `ultrabrain` | High-difficulty logic tasks |
| `deep` | Deep autonomous problem solving |
| `artistry` | Creative problem solving |
| `quick` | Simple single-file modifications |
| `unspecified-low` | Low-effort miscellaneous |
| `unspecified-high` | High-effort miscellaneous |
| `writing` | Documentation writing |

**Missing any item → that agent/category will fall back to the built-in default model → if your provider does not support that model, it will error out.**

## Verifying Configuration

After configuration, verify that all sub-agents can start normally using the following method:

```
# Test various agents in OpenCode session
> Please use explore agent to search current directory structure      # Test explore
> Please use librarian to check PyTorch documentation       # Test librarian
> Help me fix a typo using quick category        # Test quick category
> Help me analyze this code using deep category        # Test deep category
```

If any reports a model unavailable error, check whether the corresponding agent/category has the correct `model` value configured in `oh-my-opencode.json`.

## Common Errors

**Error 1**: `Model not found` or `model is not available`
→ The agent/category is missing a model override and is attempting to use the default model.

**Error 2**: `Invalid API key` or `Unauthorized`
→ The provider's baseURL or apiKey is incorrectly configured.

**Error 3**: Main agent works normally but sub-agent fails
→ `opencode.json` has the model configured but `oh-my-opencode.json` is missing the corresponding agent/category mapping.
