"""Sanity tests for the roofline/communication model.  Run: python3 -m unittest tools/test_inference_cost_model.py"""
import math
import unittest

from inference_cost_model import (Device, Link, Model, Scenario, Economics, decode_step_seconds,
                                  decode_tokens_per_s, prefill_tokens_per_s, comm_floor_tokens_per_s,
                                  usd_per_million_tokens, MODELS, LINKS)


class CostModelTests(unittest.TestCase):
    def setUp(self):
        self.dev = Device("test", mem_bw_gbs=100, peak_tops_int8=100, price_usd=1000, power_w=100,
                          mem_capacity_gb=1000, compute_efficiency=1.0)
        self.m = Model("toy", 10, 10, 1.0, layers=10, hidden=1024, kv_bytes_per_token=0)  # 10 GB weights

    def test_batch1_is_bandwidth_bound(self):
        sc = Scenario("one", self.dev)
        st = decode_step_seconds(sc, self.m, 1, 0)
        self.assertAlmostEqual(st["t_mem"], 10e9 / 100e9)          # 0.1 s
        self.assertAlmostEqual(decode_tokens_per_s(sc, self.m, 1, 0), 10.0)

    def test_large_batch_becomes_compute_bound(self):
        sc = Scenario("one", self.dev)
        st = decode_step_seconds(sc, self.m, 10000, 0)
        self.assertGreater(st["t_cmp"], st["t_mem"])

    def test_tensor_parallel_ideal_link_scales_bandwidth(self):
        ideal = Link("ideal", bw_gbs=1e9, latency_us=0.0)
        sc = Scenario("tp", self.dev, 4, ideal)
        self.assertAlmostEqual(decode_tokens_per_s(sc, self.m, 1, 0), 40.0, places=3)

    def test_tensor_parallel_latency_floor(self):
        slow = Link("slow", bw_gbs=1e9, latency_us=100.0)  # 100 us per hop
        sc = Scenario("tp", self.dev, 2, slow)
        floor = comm_floor_tokens_per_s(2, slow, self.m)
        self.assertLess(decode_tokens_per_s(sc, self.m, 1, 0), floor)
        # 2 all-reduces/layer * 10 layers * 100us * 2*log2(2) = 4 ms -> 250 tok/s floor
        self.assertAlmostEqual(floor, 250.0)

    def test_pipeline_parallel_no_gain_at_batch_1(self):
        link = Link("l", bw_gbs=1e9, latency_us=0.0)
        single = decode_tokens_per_s(Scenario("one", self.dev), self.m, 1, 0)
        pp = decode_tokens_per_s(Scenario("pp", self.dev, 4, link, "pp"), self.m, 1, 0)
        self.assertAlmostEqual(single, pp, places=6)

    def test_pipeline_parallel_scales_with_concurrency(self):
        link = Link("l", bw_gbs=1e9, latency_us=0.0)
        single = decode_tokens_per_s(Scenario("one", self.dev), self.m, 4, 0)
        pp = decode_tokens_per_s(Scenario("pp", self.dev, 4, link, "pp"), self.m, 4, 0)
        self.assertAlmostEqual(pp, 4 * single, places=6)

    def test_cost_per_token_decreases_with_batch(self):
        sc = Scenario("one", self.dev)
        econ = Economics()
        self.assertGreater(usd_per_million_tokens(sc, self.m, econ, 1, 0),
                           usd_per_million_tokens(sc, self.m, econ, 8, 0))

    def test_prefill_positive_and_above_decode(self):
        sc = Scenario("one", self.dev)
        self.assertGreater(prefill_tokens_per_s(sc, self.m), decode_tokens_per_s(sc, self.m, 1, 0))

    def test_presets_consistent(self):
        for name, m in MODELS.items():
            self.assertGreater(m.weight_bytes, 0, name)
            self.assertLessEqual(m.active_params_b, m.params_b, name)
        for name, l in LINKS.items():
            self.assertGreater(l.bw_gbs, 0, name)


if __name__ == "__main__":
    unittest.main()
