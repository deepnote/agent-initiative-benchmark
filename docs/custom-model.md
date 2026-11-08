# Custom model API

Use a trusted, importable Python module for APIs with a different format.
For a module in the current folder, run `export PYTHONPATH="$PWD"` before the CLI:

```json
{
  "provider": "custom",
  "adapter": "my_adapter:create",
  "model_id": "your-model-id",
  "max_output_tokens": 16384,
  "api_key_env": "MODEL_API_KEY"
}
```

`create(profile)` returns an object with these methods:

| Method | Contract |
| --- | --- |
| `serialize_request(request)` | Return the complete native JSON request, without credentials. Pure and deterministic. |
| `normalize_response(raw)` | Return the canonical response below. Preserve continuation state for the next request. No network access. |
| `async respond(request, *, timeout_seconds)` | Send once and return `ModelResponse(body=..., native_trace={"response": raw})`. Use `normalize_response(raw)` for the body. |
| `close()` | Optional synchronous or asynchronous cleanup. |

Import `ModelResponse` from `benchmark.runtime.responses_adapter`. Resolve keys
from the environment only when sending. Respect cancellation and the supplied
timeout. Do not retry, change prompts, shorten history or execute tools.
Construction and normalization must work without credentials or network access
so saved traces can be checked later.

Canonical requests contain `model`, `input`, `tools`, `max_output_tokens`,
`reasoning`, `store`, `parallel_tool_calls` and `include`. `input` is the complete
conversation. Function calls use `call_id`, `name` and a JSON string `arguments`.
Tool results use `type: "function_call_output"`, the matching `call_id` and a
string `output`. Preserve every message, tool definition and result.

A completed response looks like:

```json
{
  "id": "response-1",
  "status": "completed",
  "output": [
    {"type": "function_call", "call_id": "call-1", "name": "finish_task", "arguments": "{}", "status": "completed"}
  ]
}
```

Text uses `{"type": "message", "role": "assistant", "content": [{"type":
"output_text", "text": "..."}]}`. Continuations may use opaque `reasoning`
items that your serializer restores exactly. Incomplete or failed responses
must report that status. No tool from them will run.

Native responses are saved for replay. Exclude credentials and transport debug
objects. Include all model content and continuation state. The profile and
adapter module hash are pinned to the result. Keep the same module available
when checking saved runs.

[Model profiles](models.md)
