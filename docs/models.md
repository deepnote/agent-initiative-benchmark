# Models

`benchmark init` creates `benchmark.json` with one `agent` profile and two
`judges`. Set their model IDs, endpoints and key variable names. The two judge
model IDs must differ.

Each profile uses this format:

```json
{
  "provider": "responses",
  "endpoint": "https://api.openai.com/v1/responses",
  "model_id": "your-model-id",
  "api_key_env": "OPENAI_API_KEY",
  "max_output_tokens": 16384
}
```

| `provider` | API format |
| --- | --- |
| `responses` | OpenAI Responses |
| `chat_completions` | OpenAI-compatible Chat Completions |
| `anthropic` | Anthropic Messages |
| `custom` | [Your Python adapter](custom-model.md) |

Use any model ID and full endpoint supported by that API. The model must support
function tools. Set the named keys in your environment or a local `.env` beside
`benchmark.json`. Keys are not saved in run settings or authentication traces.
Use `"api_key_env": null` for a local server without authentication.

Run `benchmark check` before starting. Example standalone profiles are in
`examples/models/`. Check one with
`benchmark model check examples/models/responses.json`.
These checks validate local settings and key presence. The first live model
connection happens during `run`.

[Model options](model-options.md) covers reasoning, custom headers and output
limits. Keep settings fixed when comparing results. Another model, API protocol
or generation parameter is another experiment.

[Quickstart](quickstart.md)
