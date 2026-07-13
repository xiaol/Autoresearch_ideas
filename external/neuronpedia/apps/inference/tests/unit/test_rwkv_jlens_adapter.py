from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from neuronpedia_inference.adapters.rwkv_jlens_adapter import (
    LensChatMessage,
    LensPromptRequest,
    LensType,
    RWKVJLensAdapter,
    _format_rwkv_chat,
    _is_rwkv_generation_stop,
    _select_layers,
)
from neuronpedia_inference.adapters.fit_rwkv_jlens import DifferentiableRWKV7


class _FakeTokenizer:
    idx2token = {1: b"a", 2: b"b", 3: b"c"}

    def encode(self, text: str) -> list[int]:
        return [1] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return b"".join(self.idx2token[int(token_id)] for token_id in token_ids).decode()


class _FakeModel:
    n_layer = 3
    n_embd = 2
    vocab_size = 4
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.z = {
            "ln_out.weight": torch.ones(2),
            "ln_out.bias": torch.zeros(2),
            "head.weight": torch.tensor(
                [
                    [0.0, 2.0, 0.0, -1.0],
                    [0.0, 0.0, 2.0, -1.0],
                ]
            ),
        }

    def forward(self, token_id, state, *, collect_layers, patch):
        del token_id, state, collect_layers, patch
        hidden = [
            torch.tensor([2.0, 1.0]),
            torch.tensor([1.5, 1.0]),
            torch.tensor([1.0, 2.0]),
        ]
        return SimpleNamespace(
            logits=torch.tensor([0.0, 1.0, 2.0, -1.0]),
            state=[],
            hidden_by_layer=hidden,
        )


def _fake_adapter(jacobians: dict[int, torch.Tensor] | None = None) -> RWKVJLensAdapter:
    adapter = RWKVJLensAdapter(
        model_path="unused.pth",
        rwkv_source="unused",
        device="cpu",
        compile_cuda=False,
    )
    adapter._torch = torch
    adapter._functional = torch.nn.functional
    adapter._model = _FakeModel()
    adapter._tokenizer = _FakeTokenizer()
    adapter._jacobians = jacobians or {}
    adapter._source_means = {
        layer: torch.zeros(matrix.shape[0]) for layer, matrix in (jacobians or {}).items()
    }
    adapter._target_mean = torch.zeros(2) if jacobians else None
    adapter._jlens_status = "loaded" if jacobians else "not_found"
    adapter._jlens_error = None if jacobians else "missing test artifact"
    return adapter


class RWKVChatTemplateTest(unittest.TestCase):
    def test_checkpoint_specific_scaffold_and_history(self) -> None:
        chat = [
            LensChatMessage(role="system", content="Be concise.\n\nNo filler."),
            LensChatMessage(role="user", content="Question one.\n\nDetails."),
            LensChatMessage(role="assistant", content="First paragraph.\n\nSecond paragraph."),
            LensChatMessage(role="user", content="Question two."),
        ]
        self.assertEqual(
            _format_rwkv_chat(chat),
            "System: Be concise.\nNo filler.\n\n"
            "User: Question one.\nDetails.\n\n"
            "Assistant: First paragraph.\n\nSecond paragraph.\n\n"
            "User: Question two.\n\n"
            "Assistant: <think>\n</think",
        )
        self.assertTrue(_format_rwkv_chat(chat, enable_thinking=True).endswith("Assistant: <think"))

    def test_trailing_assistant_is_open_prefill(self) -> None:
        chat = [
            LensChatMessage(role="user", content="Continue this."),
            LensChatMessage(role="assistant", content="The answer begins"),
        ]
        self.assertEqual(
            _format_rwkv_chat(chat),
            "User: Continue this.\n\nAssistant: The answer begins",
        )

    def test_rwkv_eot_stops_before_a_new_turn(self) -> None:
        self.assertTrue(_is_rwkv_generation_stop(261, "\n\n", ["answer"]))
        self.assertTrue(_is_rwkv_generation_stop(0, "<|endoftext|>", ["answer"]))
        self.assertFalse(_is_rwkv_generation_stop(42, " answer", []))


class RWKVJacobianAdapterTest(unittest.TestCase):
    def test_select_layers_always_includes_final(self) -> None:
        self.assertEqual(_select_layers([0, 1, 2], [0], final_layer=2), [0, 2])

    def test_transport_uses_jacobian_transpose(self) -> None:
        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        adapter = _fake_adapter({0: matrix})
        adapter._source_means[0] = torch.tensor([1.0, 2.0])
        adapter._target_mean = torch.tensor([10.0, 20.0])
        hidden = torch.tensor([5.0, 7.0])
        expected = (hidden - adapter._source_means[0]) @ matrix.T + adapter._target_mean
        self.assertTrue(torch.equal(adapter.transport_hidden(hidden, 0), expected))

    def test_dual_lens_stream_has_aligned_layers_and_identity_parity(self) -> None:
        identity = torch.eye(2)
        adapter = _fake_adapter({0: identity, 1: identity})
        request = LensPromptRequest(
            model="rwkv-test",
            type=[LensType.JACOBIAN_LENS, LensType.LOGIT_LENS],
            input_token_ids=[1],
            layers=[0],
            top_n=2,
            filter_non_word_tokens=False,
            stream=True,
        )
        messages = list(adapter.run_messages(request))
        meta = messages[0]
        prompt = next(message for message in messages if message["kind"] == "prompt")
        token = next(message for message in messages if message["kind"] == "token")
        self.assertEqual(meta["types"], ["JACOBIAN_LENS", "LOGIT_LENS"])
        self.assertEqual(meta["layers_by_type"]["JACOBIAN_LENS"], [0, 2])
        self.assertEqual(meta["layers_by_type"]["LOGIT_LENS"], [0, 2])
        self.assertEqual(token["results"][0]["top_token_ids"], token["results"][1]["top_token_ids"])
        self.assertEqual(token["results"][0]["top_probs"], token["results"][1]["top_probs"])
        self.assertEqual(prompt["tokens"][0]["token_bytes"], [ord("a")])
        self.assertEqual(token["token_bytes"], [ord("a")])

    def test_dual_lens_request_does_not_silently_downgrade(self) -> None:
        adapter = _fake_adapter()
        request = LensPromptRequest(
            model="rwkv-test",
            type=[LensType.JACOBIAN_LENS, LensType.LOGIT_LENS],
            input_token_ids=[1],
        )

        messages = list(adapter.run_messages(request))

        self.assertEqual(messages[0]["kind"], "error")
        self.assertIn("JACOBIAN_LENS", messages[0]["error"])
        self.assertIn("missing test artifact", messages[0]["error"])

    def test_artifact_rejects_wrong_checkpoint_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pth"
            model_path.write_bytes(b"model bytes")
            artifact_path = Path(directory) / "lens.pt"
            torch.save(
                {
                    "J": {0: torch.eye(2)},
                    "source_means": {0: torch.zeros(2)},
                    "target_mean": torch.zeros(2),
                    "format_version": 2,
                    "n_prompts": 1,
                    "source_layers": [0],
                    "d_model": 2,
                    "n_layer": 3,
                    "architecture": "rwkv7-g1",
                    "activation_site": "block_output",
                    "transport": "affine_centered",
                    "target_layer": 2,
                    "estimator": "same_position_mean",
                    "tokenizer": "rwkv_vocab_v20230424",
                    "model_sha256": "0" * 64,
                },
                artifact_path,
            )
            adapter = _fake_adapter()
            adapter.model_path = str(model_path)
            adapter.jlens_path = str(artifact_path)
            adapter._load_jacobian_lens()
            self.assertEqual(adapter._jlens_status, "error")
            self.assertIn("model_sha256 does not match", adapter._jlens_error or "")

    def test_artifact_loads_affine_centers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_bytes = b"matching model bytes"
            model_path = Path(directory) / "model.pth"
            model_path.write_bytes(model_bytes)
            artifact_path = Path(directory) / "lens.pt"
            torch.save(
                {
                    "J": {0: torch.eye(2)},
                    "source_means": {0: torch.tensor([1.0, 2.0])},
                    "target_mean": torch.tensor([3.0, 4.0]),
                    "format_version": 2,
                    "n_prompts": 2,
                    "source_layers": [0],
                    "d_model": 2,
                    "n_layer": 3,
                    "architecture": "rwkv7-g1",
                    "activation_site": "block_output",
                    "transport": "affine_centered",
                    "target_layer": 2,
                    "estimator": "same_position_mean",
                    "tokenizer": "rwkv_vocab_v20230424",
                    "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                },
                artifact_path,
            )
            adapter = _fake_adapter()
            adapter.model_path = str(model_path)
            adapter.jlens_path = str(artifact_path)
            adapter._load_jacobian_lens()

            self.assertEqual(adapter._jlens_status, "loaded")
            self.assertEqual(adapter._jlens_n_prompts, 2)
            self.assertTrue(torch.equal(adapter._source_means[0], torch.tensor([1.0, 2.0])))
            self.assertTrue(torch.equal(adapter._target_mean, torch.tensor([3.0, 4.0])))


class RWKVSamePositionEstimatorTest(unittest.TestCase):
    def test_recurrent_values_match_while_prior_token_gradients_are_cut(self) -> None:
        def recurrence(detach_cross_position: bool):
            differentiable = object.__new__(DifferentiableRWKV7)
            differentiable.n_head = 1
            differentiable.head_size = 1
            differentiable.detach_cross_position = detach_cross_position
            values = [
                torch.ones((1, 2, 1), requires_grad=True)
                for _ in range(6)
            ]
            output = differentiable._wkv_recurrence(*values)
            gradient = torch.autograd.grad(output[0, 1, 0], values[2])[0]
            return output.detach(), gradient

        full_output, full_gradient = recurrence(False)
        cut_output, cut_gradient = recurrence(True)

        self.assertTrue(torch.allclose(cut_output, full_output))
        self.assertNotEqual(float(full_gradient[0, 0, 0]), 0.0)
        self.assertEqual(float(cut_gradient[0, 0, 0]), 0.0)
        self.assertNotEqual(float(cut_gradient[0, 1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
