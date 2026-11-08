# Model options

Set these options inside `agent` or either entry in `judges` in
`benchmark.json`, created by `benchmark init`. Each model has its own settings.

For example, the `agent` section can look like this:

```json
{
  "provider": "responses",
  "endpoint": "https://api.openai.com/v1/responses",
  "model_id": "your-model-id",
  "api_key_env": "OPENAI_API_KEY",
  "max_output_tokens": 16384,
  "reasoning_effort": "high"
}
```

Add other options alongside `model_id` and `max_output_tokens`:

| Setting | Use |
| --- | --- |
| `reasoning_effort` | Explicit effort supported by your model, such as `"high"`. Omit for the API default. |
| `parameters` | API options, such as `{"temperature": 0}` or Anthropic `{"thinking": {"type": "adaptive"}}`. |
| `auth_header`, `auth_prefix` | Defaults: `"Authorization"`, `"Bearer "`. For an API-key header use `"x-api-key"`, `""`. |
| `headers_env` | Extra headers from environment variables, such as `{"anthropic-beta": "MODEL_BETA_FLAGS"}`. |
| `api_version` | Anthropic API version. Default: `"2026-10-21"`. |
| `instruction_role` | Chat instruction role: `"system"` by default, or `"developer"`. |
| `token_limit_field` | Chat output limit: `"max_completion_tokens"` by default, or `"max_tokens"`. |
| `judge_parallel_reads` | Allow grouped judge reads. Default: `true`. Both judges must use the same value. Agent grouping follows the task protocol. |
| `max_response_bytes` | Complete HTTP response bound. Default: 128 MiB. Oversized responses fail. |

`max_output_tokens` is required. Other settings use the API default unless listed
above. `parameters` cannot replace messages, tools, storage, tool selection or
benchmark limits. Unsupported API options fail without a fallback. HTTP errors
are recorded without automatic retries. Conversations are sent in full.

Keep settings fixed when comparing results. Changing the model, API protocol,
reasoning, output limit or other generation options changes the experiment.

[Model setup](models.md)
