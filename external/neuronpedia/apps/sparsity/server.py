"""
FastAPI server for analyzing MLP neuron connections in sparse circuit models.
"""

import os
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from transformers import AutoModelForCausalLM

# Load .env file if present
load_dotenv()

MODEL_HF_ID = "openai/circuit-sparsity"
SECRET = os.environ.get("SECRET")

# Global model reference
model = None


def load_model():
    """Load the model and move to appropriate device."""
    print(f"Loading model: {MODEL_HF_ID}")
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_HF_ID, trust_remote_code=True, torch_dtype="auto"
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    m = m.to(device).eval()
    print(f"Model loaded on {device}")
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    model = load_model()
    yield
    # Cleanup if needed
    model = None


async def verify_secret(x_secret_key: str | None = Header(default=None)):
    """Verify the secret key header if SECRET is configured."""
    if SECRET is not None:
        if x_secret_key is None:
            raise HTTPException(status_code=401, detail="X-SECRET-KEY header required")
        if x_secret_key != SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret key")


app = FastAPI(
    title="Sparse Circuit Analyzer",
    description="Analyze MLP neuron connections in sparse circuit models",
    lifespan=lifespan,
    dependencies=[Depends(verify_secret)],
)


def get_layers():
    """Get transformer layers from model."""
    return model.circuit_model.transformer["h"]


def neuron_get_connected_reschannels(layer_idx: int, neuron_idx: int):
    """Get residual stream channels connected to a specific MLP neuron."""
    layers = get_layers()
    c_fc_weights = layers[layer_idx].mlp.c_fc.weight[neuron_idx, :].tolist()
    c_proj_weights = layers[layer_idx].mlp.c_proj.weight[:, neuron_idx].tolist()

    return {
        "in": [
            {"channel_id": i, "weight": w} for i, w in enumerate(c_fc_weights) if w != 0
        ],
        "out": [
            {"channel_id": i, "weight": w}
            for i, w in enumerate(c_proj_weights)
            if w != 0
        ],
    }


def find_neurons_reading_channel(layer_idx: int, channel_id: int, top_k: int = 5):
    """Find neurons in a layer that read from a specific channel (via c_fc)."""
    layers = get_layers()
    c_fc_weights = layers[layer_idx].mlp.c_fc.weight[:, channel_id].tolist()

    neurons = [
        {"neuron_id": i, "weight": w} for i, w in enumerate(c_fc_weights) if w != 0
    ]
    neurons.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return neurons[:top_k]


def find_neurons_writing_channel(layer_idx: int, channel_id: int, top_k: int = 5):
    """Find neurons in a layer that write to a specific channel (via c_proj)."""
    layers = get_layers()
    c_proj_weights = layers[layer_idx].mlp.c_proj.weight[channel_id, :].tolist()

    neurons = [
        {"neuron_id": i, "weight": w} for i, w in enumerate(c_proj_weights) if w != 0
    ]
    neurons.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return neurons[:top_k]


def trace_circuit_forward(
    start_layer: int, start_neuron: int, depth: int = 3, top_k: int = 3
):
    """Trace a circuit forward from a starting neuron."""
    layers = get_layers()
    num_layers = len(layers)

    def trace_from_neuron(layer: int, neuron: int, remaining_depth: int):
        if remaining_depth == 0 or layer >= num_layers - 1:
            return None

        connections = neuron_get_connected_reschannels(layer, neuron)
        out_channels = sorted(
            connections["out"], key=lambda x: abs(x["weight"]), reverse=True
        )[:top_k]

        result = []
        for ch in out_channels:
            channel_id = ch["channel_id"]
            write_weight = ch["weight"]

            for next_layer in range(layer + 1, num_layers):
                readers = find_neurons_reading_channel(
                    next_layer, channel_id, top_k=top_k
                )
                for reader in readers:
                    result.append(
                        {
                            "layer": next_layer,
                            "neuron": reader["neuron_id"],
                            "read_weight": reader["weight"],
                            "via_channel": channel_id,
                            "write_weight": write_weight,
                            "children": trace_from_neuron(
                                next_layer, reader["neuron_id"], remaining_depth - 1
                            ),
                        }
                    )

        result.sort(
            key=lambda x: abs(x["write_weight"]) * abs(x["read_weight"]), reverse=True
        )
        return result[: top_k * 2] if result else None

    return trace_from_neuron(start_layer, start_neuron, depth) or []


def trace_circuit_backward(
    start_layer: int, start_neuron: int, depth: int = 3, top_k: int = 3
):
    """Trace a circuit backward from a starting neuron."""

    def trace_from_neuron(layer: int, neuron: int, remaining_depth: int):
        if remaining_depth == 0 or layer <= 0:
            return None

        connections = neuron_get_connected_reschannels(layer, neuron)
        in_channels = sorted(
            connections["in"], key=lambda x: abs(x["weight"]), reverse=True
        )[:top_k]

        result = []
        for ch in in_channels:
            channel_id = ch["channel_id"]
            read_weight = ch["weight"]

            for prev_layer in range(layer - 1, -1, -1):
                writers = find_neurons_writing_channel(
                    prev_layer, channel_id, top_k=top_k
                )
                for writer in writers:
                    result.append(
                        {
                            "layer": prev_layer,
                            "neuron": writer["neuron_id"],
                            "write_weight": writer["weight"],
                            "via_channel": channel_id,
                            "read_weight": read_weight,
                            "parents": trace_from_neuron(
                                prev_layer, writer["neuron_id"], remaining_depth - 1
                            ),
                        }
                    )

        result.sort(
            key=lambda x: abs(x["write_weight"]) * abs(x["read_weight"]), reverse=True
        )
        return result[: top_k * 2] if result else None

    return trace_from_neuron(start_layer, start_neuron, depth) or []


@app.get("/")
async def root():
    """Health check and model info."""
    layers = get_layers()
    return {
        "status": "ok",
        "model": MODEL_HF_ID,
        "num_layers": len(layers),
        "mlp_size": layers[0].mlp.c_fc.weight.shape[0],
        "d_model": layers[0].mlp.c_proj.weight.shape[0],
    }


@app.get("/neuron/{layer}/{neuron}")
async def get_neuron_connections(
    layer: int,
    neuron: int,
    trace_depth: int = Query(default=2, description="Depth of circuit trace"),
    trace_k: int = Query(
        default=3, description="Top K channels/neurons per step in trace"
    ),
):
    """
    Get circuit traces for a specific MLP neuron.

    Returns forward trace (downstream neurons) and backward trace (upstream neurons).
    """
    layers = get_layers()
    num_layers = len(layers)

    if layer < 0 or layer >= num_layers:
        raise HTTPException(
            status_code=400, detail=f"Layer must be between 0 and {num_layers - 1}"
        )

    mlp_size = layers[layer].mlp.c_fc.weight.shape[0]
    if neuron < 0 or neuron >= mlp_size:
        raise HTTPException(
            status_code=400, detail=f"Neuron index must be between 0 and {mlp_size - 1}"
        )

    return {
        "layer": layer,
        "neuron": neuron,
        "trace_forward": trace_circuit_forward(
            layer, neuron, depth=trace_depth, top_k=trace_k
        ),
        "trace_backward": trace_circuit_backward(
            layer, neuron, depth=trace_depth, top_k=trace_k
        ),
    }


@app.get("/channel/{channel_id}")
async def get_channel_connections(
    channel_id: int,
    top_k: int = Query(
        default=10, description="Number of top neurons to return per layer"
    ),
):
    """
    Get all neurons that read from or write to a specific residual channel.
    """
    layers = get_layers()
    num_layers = len(layers)
    d_model = layers[0].mlp.c_proj.weight.shape[0]

    if channel_id < 0 or channel_id >= d_model:
        raise HTTPException(
            status_code=400, detail=f"Channel must be between 0 and {d_model - 1}"
        )

    readers_by_layer = {}
    writers_by_layer = {}

    for layer_idx in range(num_layers):
        readers = find_neurons_reading_channel(layer_idx, channel_id, top_k=top_k)
        writers = find_neurons_writing_channel(layer_idx, channel_id, top_k=top_k)

        if readers:
            readers_by_layer[layer_idx] = readers
        if writers:
            writers_by_layer[layer_idx] = writers

    return {
        "channel_id": channel_id,
        "readers_by_layer": readers_by_layer,
        "writers_by_layer": writers_by_layer,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5005)
