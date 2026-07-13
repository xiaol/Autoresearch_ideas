# Sparse Circuit Analyzer

FastAPI server for analyzing MLP neuron connections in [openai/circuit-sparsity](https://github.com/openai/circuit_sparsity) models.

## Setup

```bash
uv sync
```

## Run

```bash
uv run uvicorn server:app --port 5005 --host 0.0.0.0 --reload
```

Optional: set `SECRET` in `.env` to require `X-SECRET-KEY` header.

## Endpoints

- `GET /` — health check, model info
- `GET /neuron/{layer}/{neuron}` — circuit traces for a neuron
  - `?trace_depth=2` — trace depth
  - `?trace_k=3` — top K channels/neurons per step
- `GET /channel/{channel_id}` — neurons connected to a channel

## Example

```bash
curl "http://localhost:8000/neuron/2/1717?trace_k=2"
```
