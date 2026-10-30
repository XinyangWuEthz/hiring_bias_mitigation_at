"""CPU checks for global counts, policy masking and observation-only hooks."""

import copy
import importlib.util
import json
import math
import os
import random
import socket
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from hiring_bias_mitigation.mitigation import grpo_diagnostics as D


def completion(index, *, side="hire", correct=True, advantage=0.0, weight=1.0,
               zero_std=True, format_reward=0.2, mask=(1, 1), threshold=False):
    return {
        "prompt_id": f"prompt-{index}", "reference_decision": side,
        "decision": side if correct else ("reject" if side == "hire" else "hire"),
        "winning_votes": 5, "decided_draws": 7,
        "decision_reward": float(correct), "format_reward": format_reward,
        "reward_zero_std": zero_std, "advantage": advantage,
        "is_threshold_masked": threshold,
        "token_counts": D.token_diagnostics(mask, [-1.0] * len(mask), [-1.0] * len(mask),
                                             [weight], advantage, threshold_masked=threshold),
    }


class CountTests(unittest.TestCase):
    def test_rank_and_window_weighting_with_absent_class(self):
        window = D.CountWindow(["reject/5_of_7"])
        # First global rollout: rank0 has one hire; rank1 has three rejections.
        window.add_samples([completion(0)] + [completion(0, correct=False)] * 3)
        # Different denominator in the second rollout: both are hires.
        window.add_samples([completion(1)] * 2)
        result = window.snapshot()
        self.assertEqual(result["basic"]["sampled_count_ref_hire"], 6)
        self.assertEqual(result["basic"]["sampled_hire_count_ref_hire"], 3)
        self.assertEqual(result["basic"]["sampled_hire_rate_ref_hire"], 0.5)
        self.assertNotEqual(result["basic"]["sampled_hire_rate_ref_hire"], (0.25 + 1) / 2)
        self.assertIsNone(result["basic"]["sampled_hire_rate_ref_reject"])
        empty = result["strata"]["reject/5_of_7"]
        self.assertEqual(empty["counts"]["completion_tokens"], 0)
        self.assertEqual(set(empty["counts"]), set(D.COUNT_KEYS))
        self.assertTrue(all(value is None for value in empty["rates"].values()))

    def test_group_crosses_four_rank_boundaries(self):
        rows = [completion(0, correct=i < 8, advantage=1 if i < 8 else -1,
                           zero_std=False) for i in range(16)]
        rank_shards = [rows[i:i + 4] for i in range(0, 16, 4)]
        global_rows = [row for shard in rank_shards for row in shard]
        result = D.group_counts(global_rows)["hire/5_of_7"]
        self.assertEqual(result["groups"], 1)
        self.assertEqual(result["mixed_decision_groups"], 1)
        self.assertEqual(result["effective_adv_groups"], 1)
        with self.assertRaisesRegex(ValueError, "whole number"):
            D.group_counts(rank_shards[0])

    def test_correct_mixed_wrong_and_format_only_groups(self):
        rows = [completion(0)] * 16 + [completion(1, correct=False)] * 16
        rows += [completion(2, correct=i < 8, advantage=1 if i < 8 else -1,
                            zero_std=False) for i in range(16)]
        rows += [completion(3, correct=False, advantage=1 if i < 8 else -1,
                            zero_std=False, format_reward=0.2 if i < 8 else 0.0)
                 for i in range(16)]
        result = D.group_counts(rows)["hire/5_of_7"]
        self.assertEqual(result["all_correct_groups"], 1)
        self.assertEqual(result["all_wrong_groups"], 2)
        self.assertEqual(result["mixed_decision_groups"], 1)
        self.assertEqual(result["reward_zero_std_groups"], 2)
        self.assertEqual(result["effective_adv_groups"], 2)
        self.assertEqual(result["format_only_effective_groups"], 1)

    def test_threshold_mask_empty_and_underflow_are_distinct(self):
        masked = [completion(0, correct=i < 8, advantage=1 if i < 8 else -1,
                             zero_std=False, weight=0, threshold=True) for i in range(16)]
        empty = [completion(1, advantage=1, zero_std=False, mask=(0, 0))] * 16
        underflow = [completion(2, advantage=1, zero_std=False, weight=0)] * 16
        result = D.group_counts(masked + empty + underflow)["hire/5_of_7"]
        self.assertEqual(result["fully_lost_effective_adv_groups"], 3)
        self.assertEqual(result["fully_zero_weighted_eligible_adv_groups"], 2)
        self.assertEqual(result["fully_threshold_masked_eligible_adv_groups"], 1)
        self.assertEqual(result["empty_only_effective_adv_groups"], 1)
        self.assertEqual(result["is_threshold_masked_completions"], 16)
        self.assertEqual(result["policy_zero_weight_completions"], 32)

    def test_tokens_padding_nan_and_zero_weight(self):
        result = D.token_diagnostics([1, 1, 0], [-1, -2, 999], [-1.5, math.nan, 999],
                                     [0], 1, threshold_masked=True)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["policy_zero_weight_tokens"], 2)
        self.assertEqual(result["scored_logprob_tokens"], 1)
        self.assertEqual(result["missing_sampling_logprob_tokens"], 1)
        self.assertEqual(result["abs_logprob_difference_sum"], 0.5)
        self.assertEqual(result["retained_effective_adv_completions"], 0)

    def test_metadata_mismatch_is_not_silently_regrouped(self):
        rows = [completion(0)] * 15 + [completion(1)]
        with self.assertRaisesRegex(ValueError, "different prompt"):
            D.group_counts(rows)

    def test_vote_denominator_remains_in_stratum(self):
        a, b = completion(0), completion(1)
        b["decided_draws"] = 8
        result = D.group_counts([a] * 16 + [b] * 16)
        self.assertEqual(set(result), {"hire/5_of_7", "hire/5_of_8"})


class Array:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)


class GRPOTrainer:
    def __init__(self):
        self.model = SimpleNamespace(training=True)
        self.accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        self.state = SimpleNamespace(global_step=425)
        self.args = SimpleNamespace(
            scale_rewards="group", multi_objective_aggregation="sum_then_normalize",
            mask_truncated_completions=False, off_policy_mask_threshold=None,
            top_entropy_quantile=1.0, loss_type="dapo")
        self.num_generations = self.num_generations_eval = 16
        self.use_vllm = self.vllm_importance_sampling_correction = True
        self.vllm_importance_sampling_mode = "sequence_mask"
        self.vllm_importance_sampling_clip_max = 3.0
        self.vllm_importance_sampling_clip_min = None
        self.tools = self.environments = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.reward_result = object()
        self.logged = []
        self.result = {key: Array(value) for key, value in {
            "completion_mask": [[1, 1]] * 16, "old_per_token_logps": [[-1, -1]] * 16,
            "sampling_per_token_logps": [[-1, -1]] * 16,
            "importance_sampling_ratio": [[1]] * 16, "advantages": [0.0] * 16,
        }.items()}

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["sampled_hire_rate_ref_hire"].append(0.123)
        return self.reward_result

    def _generate_and_score_completions(self, inputs):
        self._calculate_rewards(
            inputs, ["prompt"] * len(inputs), [r["decision"] for r in inputs], [])
        return self.result

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        # Reproduce TRL.log's overwrite behavior, not only a passive logging mock.
        for key, values in self._metrics[mode].items():
            logs[("eval_" if mode == "eval" else "") + key] = sum(values) / len(values)
        self.logged.append(copy.deepcopy(logs))
        self._metrics[mode].clear()


class HookTests(unittest.TestCase):
    def install(self, trainer, path, **kwargs):
        return D.install(trainer, parse_decision=lambda text, lang: text, output_dir=path,
                         gather=lambda payload: payload["rows"] if not payload["error"] else
                         (_ for _ in ()).throw(ValueError(payload["error"])), **kwargs)

    def test_default_basic_hook_overrides_old_means_and_preserves_outputs_rng(self):
        trainer = GRPOTrainer()
        with tempfile.TemporaryDirectory() as path:
            self.install(trainer, path)
            before_rng = random.getstate()
            inputs = [completion(0)] + [completion(0, correct=False)] * 3
            actual = trainer._calculate_rewards(inputs, [], [r["decision"] for r in inputs], [])
            self.assertIs(actual, trainer.reward_result)
            trainer._calculate_rewards([completion(1)] * 2, [], ["hire"] * 2, [])
            trainer.log({"loss": 0.03})
            self.assertEqual(random.getstate(), before_rng)
            self.assertEqual(trainer.logged[-1]["sampled_hire_rate_ref_hire"], 0.5)
            self.assertNotIn("sampled_hire_rate_ref_reject", trainer.logged[-1])
            event = json.loads((Path(path) / "rollout_diagnostics.jsonl").read_text())
            self.assertIsNone(event["basic"]["sampled_hire_rate_ref_reject"])
            self.assertEqual(event["basic"]["sampled_count_ref_hire"], 6)
            trainer.log({"loss": 0.02})
            self.assertNotIn("sampled_count_ref_hire", trainer.logged[-1])

    def test_detailed_hook_returns_identical_tensors_and_logs_counts(self):
        trainer = GRPOTrainer()
        rows = [completion(0)] * 16
        original = copy.deepcopy({key: value.values for key, value in trainer.result.items()})
        def reward_snapshot(trainer, result, k):
            return [{"decision_reward": 1.0, "format_reward": 0.2,
                     "reward_zero_std": True}] * 16

        def sampling_snapshot(trainer, result):
            return [{"is_threshold_masked": False,
                     "is_zero_without_threshold": False}] * 16
        with tempfile.TemporaryDirectory() as path:
            self.install(trainer, path, detailed=True, reward_snapshot=reward_snapshot,
                         sampling_snapshot=sampling_snapshot)
            before_rng = random.getstate()
            result = trainer._generate_and_score_completions(rows)
            self.assertIs(result, trainer.result)
            self.assertEqual(original, {key: value.values for key, value in trainer.result.items()})
            self.assertEqual(random.getstate(), before_rng)
            trainer.log({})
            event = json.loads((Path(path) / "rollout_diagnostics.jsonl").read_text())
            self.assertEqual(event["total"]["counts"]["groups"], 1)
            self.assertEqual(event["total"]["counts"]["reward_zero_std_groups"], 1)
            self.assertTrue(event["policy_mask_excludes_kl"])

    def test_dapo_basic_counts_actual_internal_reward_rows(self):
        class DAPOTrainer(GRPOTrainer):
            def _generate_and_score_completions(self, unused_outer_inputs):
                for size in (16, 32):
                    rows = [completion(0, side="reject")] * size
                    self._calculate_rewards(rows, [], ["reject"] * size, [])
                return self.result

        trainer = DAPOTrainer()
        with tempfile.TemporaryDirectory() as path:
            self.install(trainer, path)
            trainer._generate_and_score_completions([completion(0)])
            trainer.log({})
            self.assertEqual(trainer.logged[-1]["sampled_count_ref_reject"], 48)
            self.assertEqual(trainer.logged[-1]["sampled_hire_rate_ref_reject"], 0)
        with self.assertRaisesRegex(ValueError, "GRPOTrainer only"):
            self.install(DAPOTrainer(), "/unused", detailed=True)

    def test_detailed_eval_uses_eval_group_size_and_its_own_window(self):
        trainer = GRPOTrainer()
        trainer.model.training = False
        trainer.num_generations_eval = 8
        rows = [completion(1, side="reject", correct=False)] * 8
        rows += [completion(2, correct=i < 4, advantage=1 if i < 4 else -1,
                            zero_std=False) for i in range(8)]
        trainer.result["advantages"] = Array([0.0] * 8 + [1.0] * 4 + [-1.0] * 4)
        observed_group_sizes = []

        def reward_snapshot(trainer, result, k):
            observed_group_sizes.append(k)
            return [{"decision_reward": row["decision_reward"],
                     "format_reward": row["format_reward"],
                     "reward_zero_std": row["reward_zero_std"]} for row in rows]

        def sampling_snapshot(trainer, result):
            return [{"is_threshold_masked": False, "is_zero_without_threshold": False}] * 16
        with tempfile.TemporaryDirectory() as path:
            self.install(trainer, path, detailed=True, reward_snapshot=reward_snapshot,
                         sampling_snapshot=sampling_snapshot,
                         strata=("hire/5_of_7", "reject/5_of_7"))
            original = trainer.result
            self.assertIs(trainer._generate_and_score_completions(rows), original)
            trainer.log({"eval_loss": 0.2})
            self.assertEqual(observed_group_sizes, [8])
            self.assertEqual(trainer.logged[-1]["eval_rollout/all/groups"], 2)
            self.assertEqual(
                trainer.logged[-1]["eval_rollout/reject/5_of_7/all_wrong_groups"], 1)
            self.assertEqual(
                trainer.logged[-1]["eval_rollout/hire/5_of_7/mixed_decision_groups"], 1)
            event = json.loads((Path(path) / "rollout_diagnostics.jsonl").read_text())
            self.assertEqual(event["mode"], "eval")
            self.assertEqual(event["total"]["counts"]["completions"], 16)
            trainer.model.training = True
            trainer.log({"train_runtime": 1.0})
            self.assertNotIn("rollout/all/groups", trainer.logged[-1])

    def test_other_masks_are_rejected(self):
        trainer = GRPOTrainer()
        trainer.args.off_policy_mask_threshold = 0.1
        with self.assertRaisesRegex(ValueError, "other policy masks"):
            self.install(trainer, "/unused", detailed=True)


def distributed_worker(rank, port, directory):
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE="4", LOCAL_WORLD_SIZE="4")
    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=True)
    sizes = [4, 12, 8, 8]
    side = "reject" if rank < 2 else "hire"
    rows = [completion(0 if rank < 2 else 1, side=side,
                       correct=(rank == 1 or rank == 2 or (rank == 3 and i < 4)),
                       advantage=1, zero_std=False) for i in range(sizes[rank])]
    global_rows = D._gather_payload({"rows": rows, "error": None})
    groups = D.group_counts(global_rows)
    assert groups["reject/5_of_7"]["groups"] == groups["hire/5_of_7"]["groups"] == 1
    assert [row["reference_decision"] for row in global_rows] == ["reject"] * 16 + ["hire"] * 16
    trainer = GRPOTrainer()
    trainer.accelerator = accelerator
    D.install(trainer, parse_decision=lambda text, lang: text, output_dir=directory)
    trainer._calculate_rewards(rows, [], [row["decision"] for row in rows], [])
    trainer.log({})
    observed = trainer.logged[-1]
    assert observed["sampled_hire_rate_ref_reject"] == 0.25
    assert observed["sampled_hire_rate_ref_hire"] == 0.75
    all_observed = [None] * 4
    dist.all_gather_object(all_observed, observed)
    assert all(value == all_observed[0] for value in all_observed)
    # All ranks participate in an error-bearing gather and all see the same failure.
    try:
        D._gather_payload({"rows": [], "error": "rank2 validation" if rank == 2 else None})
    except ValueError as exc:
        assert "rank2 validation" in str(exc)
    else:
        raise AssertionError("remote rank validation failure was not propagated")
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("trl"),
                     "installed PyTorch/TRL CPU integration is run in the Daint environment")
class InstalledTensorTests(unittest.TestCase):
    def test_four_process_gloo_order_counts_zero_class_and_rank_errors(self):
        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(port, directory), nprocs=4, join=True)
            events = (Path(directory) / "rollout_diagnostics.jsonl").read_text().splitlines()
            self.assertEqual(len(events), 1)  # Only rank zero writes.
            self.assertEqual(json.loads(events[0])["basic"]["sampled_count_ref_hire"], 16)

    def test_installed_reward_and_log_count_windows_without_model_forward(self):
        # Accelerate uses process-wide shared state. Other Trainer tests may already
        # have selected CUDA, so Accelerator(cpu=True) alone cannot isolate this test.
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", ACCELERATE_USE_CPU="true",
                   ACCELERATE_TORCH_DEVICE="cpu", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR", "MASTER_PORT"):
            env.pop(key, None)
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        script = (
            "import runpy, sys; "
            "ns = runpy.run_path(sys.argv[1], run_name='isolated_grpo_diagnostics_test'); "
            "case = ns['InstalledTensorTests']("
            "'test_installed_reward_and_log_count_windows_without_model_forward'); "
            "case._check_installed_reward_and_log_count_windows()"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve())],
            env=env, capture_output=True, text=True, timeout=240, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def _check_installed_reward_and_log_count_windows(self):
        import torch
        from accelerate import Accelerator
        from transformers import TrainerControl, TrainerState
        from trl import GRPOTrainer as InstalledTrainer

        trainer = InstalledTrainer.__new__(InstalledTrainer)
        trainer.__dict__.update(GRPOTrainer().__dict__)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.accelerator = Accelerator(cpu=True)
        self.assertEqual(trainer.accelerator.device.type, "cpu")
        self.assertEqual(trainer.accelerator.num_processes, 1)
        trainer.state = TrainerState(global_step=425)
        trainer.control = TrainerControl()
        trainer.callback_handler = SimpleNamespace(
            on_log=lambda args, state, control, logs: control)
        trainer.args.report_to = []
        trainer.args.include_num_input_tokens_seen = "no"
        trainer.log_completions = False

        def decision_reward(completions, reference_decision, **kwargs):
            return [float(c == r) for c, r in zip(completions, reference_decision, strict=True)]

        def format_reward(completions, **kwargs):
            return [0.2] * len(completions)

        trainer.reward_funcs = [decision_reward, format_reward]
        trainer.reward_func_names = ["decision_reward", "format_reward"]
        trainer.reward_processing_classes = [None, None]
        trainer.reward_weights = torch.ones(2)
        with tempfile.TemporaryDirectory() as path:
            D.install(trainer, parse_decision=lambda text, lang: text, output_dir=path)
            rng = torch.random.get_rng_state().clone()
            for decisions in (["hire", "reject", "reject", "reject"], ["hire", "hire"]):
                rows = [{"reference_decision": "hire"}] * len(decisions)
                actual = trainer._calculate_rewards(rows, ["p"] * len(rows), decisions, [])
                torch.testing.assert_close(
                    actual[:, 0], torch.tensor([float(d == "hire") for d in decisions]))
            trainer._metrics["train"]["sampled_hire_rate_ref_hire"] = [0.25, 1.0]
            trainer.log({"loss": 0.03})
            self.assertEqual(trainer.state.log_history[-1]["sampled_hire_rate_ref_hire"], 0.5)
            self.assertEqual(trainer.state.log_history[-1]["sampled_count_ref_hire"], 6)
            self.assertNotIn("sampled_hire_rate_ref_reject", trainer.state.log_history[-1])
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            trainer.model.eval()
            trainer._calculate_rewards([{"reference_decision": "reject"}], ["p"], ["hire"], [])
            trainer.log({"eval_loss": 0.04})
            self.assertEqual(
                trainer.state.log_history[-1]["eval_sampled_hire_rate_ref_reject"], 1.0)
            self.assertEqual(trainer.state.log_history[-1]["eval_sampled_count_ref_hire"], 0)
            trainer.model.train()
            trainer.log({"train_runtime": 1.0})
            self.assertNotIn("sampled_count_ref_hire", trainer.state.log_history[-1])
            events = [json.loads(line) for line in
                      (Path(path) / "rollout_diagnostics.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 2)
            self.assertIsNone(events[0]["basic"]["sampled_hire_rate_ref_reject"])

    def test_actual_sequence_threshold_mask_and_underflow(self):
        import torch

        trainer = SimpleNamespace(vllm_importance_sampling_clip_max=3.0)
        old = torch.tensor([[0.0, 0.0], [0.0, 0.0], [-1000.0, 0.0], [0.0, 0.0]])
        sample = torch.tensor([[-0.7, -0.7], [-0.1, -0.1], [0.0, 0.0], [math.nan, 0.0]])
        mask = torch.ones_like(old)
        raw = torch.exp(torch.nan_to_num((old - sample) * mask, nan=0).sum(-1, keepdim=True))
        actual = raw.masked_fill(raw > 3, 0)
        result = {"completion_mask": mask, "old_per_token_logps": old,
                  "sampling_per_token_logps": sample, "importance_sampling_ratio": actual}
        before = {key: value.clone() for key, value in result.items()}
        snapshot = D._sampling_snapshot(trainer, result)
        self.assertEqual(
            [row["is_threshold_masked"] for row in snapshot], [True, False, False, False])
        self.assertEqual(
            [row["is_zero_without_threshold"] for row in snapshot], [False, False, True, False])
        for key, value in before.items():
            torch.testing.assert_close(result[key], value, equal_nan=True)

    def test_actual_float32_reward_zero_std(self):
        import torch

        trainer = SimpleNamespace(reward_func_names=["decision_reward", "format_reward"],
                                  reward_weights=torch.ones(2))
        rewards = torch.tensor([[1, 0.2]] * 16 + [[0, 0.2]] * 8 + [[1, 0.2]] * 8)
        snapshot = D._reward_snapshot(trainer, rewards, 16)
        self.assertTrue(all(row["reward_zero_std"] for row in snapshot[:16]))
        self.assertTrue(all(not row["reward_zero_std"] for row in snapshot[16:]))
        singleton = D._reward_snapshot(trainer, rewards[:2], 1)
        self.assertTrue(all(row["reward_zero_std"] for row in singleton))


if __name__ == "__main__":
    unittest.main()
