"""Real four-process CPU reproducer for the pilot's output-directory startup race."""

import multiprocessing
import queue
import tempfile
import unittest
from pathlib import Path

from hiring_bias_mitigation.mitigation import grpo_continuation as G


class _CpuAccelerator:
    """Exercise the production barrier protocol using CPU process synchronization."""

    def __init__(self, rank, barrier, main_waiting):
        self.is_main_process = rank == 0
        self.barrier, self.main_waiting = barrier, main_waiting

    def wait_for_everyone(self):
        if self.is_main_process:
            self.main_waiting.set()
        self.barrier.wait(timeout=15)


def _worker(rank, fixed, output, source, checked, initial_three, all_four,
            release_late_rank, main_waiting, metadata_written, writes, results,
            errors, writer_fails):
    try:
        if rank == 3 and not release_late_rank.wait(timeout=15):
            raise TimeoutError("late-rank release missing")
        contract = {"output": Path(output), "source": Path(source)}
        selected = G.choose_resume(contract, None)
        with checked.get_lock():
            checked.value |= 1 << rank

        def write_tracking():
            if fixed:
                with checked.get_lock():
                    assert checked.value == 15, "metadata written before all resume checks"
            if writer_fails:
                raise OSError("simulated metadata write failure")
            Path(output, "wandb_run_id").write_text("run-id00\n")
            with writes.get_lock():
                writes.value += 1
            metadata_written.set()

        def broadcast_error(error):
            if rank == 0:
                for _ in range(3):
                    errors.put(error)
                return error
            return errors.get(timeout=15)

        if fixed:
            accelerator = _CpuAccelerator(rank, all_four, main_waiting)
            G.initialize_tracking_after_resume(accelerator, write_tracking, broadcast_error)
            assert Path(output, "wandb_run_id").is_file()
        else:
            # Match the observed job: 0/1/2 choose an empty output, rank 0 writes
            # the W&B ID, then rank 3 reaches its empty-directory check.
            if rank < 3:
                initial_three.wait(timeout=15)
            if rank == 0:
                write_tracking()
                release_late_rank.set()
            if not metadata_written.wait(timeout=15):
                raise TimeoutError("tracking metadata missing")
        results.put({"rank": rank, "selected": selected, "error": None})
    except Exception as exc:
        results.put({"rank": rank, "selected": None, "error": f"{type(exc).__name__}: {exc}"})


class ContinuationInitializationTests(unittest.TestCase):
    def _four_ranks(self, fixed, writer_fails=False):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, source = root / "pilot", root / "source" / "checkpoint-425"
            output.mkdir()
            source.mkdir(parents=True)
            checked, writes = context.Value("i", 0), context.Value("i", 0)
            initial_three, all_four = context.Barrier(3), context.Barrier(4)
            release_late = context.Event()
            main_waiting, metadata_written = context.Event(), context.Event()
            results = context.Queue()
            errors = context.Queue()
            processes = [context.Process(target=_worker, args=(
                rank, fixed, str(output), str(source), checked, initial_three, all_four,
                release_late, main_waiting, metadata_written, writes, results,
                errors, writer_fails)) for rank in range(4)]
            for process in processes:
                process.start()
            try:
                if fixed:
                    self.assertTrue(main_waiting.wait(timeout=15), "rank 0 never entered barrier")
                    self.assertFalse(metadata_written.is_set())
                    self.assertFalse((output / "wandb_run_id").exists())
                    release_late.set()
                records = []
                for _ in processes:
                    try:
                        records.append(results.get(timeout=20))
                    except queue.Empty:
                        self.fail("worker did not report a result")
                for process in processes:
                    process.join(timeout=5)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual(writes.value, 0 if writer_fails else 1)
                return sorted(records, key=lambda row: row["rank"]), str(source)
            finally:
                release_late.set()
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
                results.close()
                results.join_thread()
                errors.close()
                errors.join_thread()

    def test_old_order_deterministically_reproduces_job_5130709(self):
        records, source = self._four_ranks(fixed=False)
        self.assertTrue(all(row["selected"] == source for row in records[:3]))
        self.assertIn("output is not empty", records[3]["error"])

    def test_barrier_keeps_output_empty_until_all_four_choose_the_source(self):
        records, source = self._four_ranks(fixed=True)
        self.assertTrue(all(row["error"] is None for row in records), records)
        self.assertTrue(all(row["selected"] == source for row in records))

    def test_failed_attempt_and_partial_training_files_still_rejected(self):
        filenames = ("wandb_run_id", "run_config.json", "continuation_rank0.jsonl", "partial.bin")
        for filename in filenames:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                (output / filename).write_text("existing evidence")
                with self.assertRaisesRegex(ValueError, "output is not empty"):
                    G.choose_resume({"output": output, "source": output / "source"}, None)
                self.assertEqual((output / filename).read_text(), "existing evidence")

    def test_rank_zero_write_failure_reaches_all_four_ranks_without_hanging(self):
        records, _ = self._four_ranks(fixed=True, writer_fails=True)
        self.assertTrue(all("simulated metadata write failure" in row["error"]
                            for row in records), records)


if __name__ == "__main__":
    unittest.main()
