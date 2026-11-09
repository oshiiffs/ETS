import tempfile
import unittest
from pathlib import Path

from src.predict import load_model, predict_service_time, predict_waiting_time
from src.queue_simulator import simulate_department_wait_seconds


class DepartmentModelTests(unittest.TestCase):
    def test_supported_and_unsupported_departments(self):
        with self.assertRaises(ValueError): predict_service_time("UNKNOWN", model=object())

    def test_mixed_department_fifo_uses_each_estimate(self):
        students=[{"department":"CCS"},{"department":"CBM"}]
        seen=[]
        def duration(s):
            seen.append(s["department"])
            return {"CCS":60,"CBM":120}[s["department"]]
        wait=simulate_department_wait_seconds(students,1,[],duration)
        self.assertEqual(seen,["CCS","CBM"])
        self.assertEqual(wait,180)

    def test_idle_window_and_busy_residual(self):
        d=lambda s: 90
        self.assertEqual(simulate_department_wait_seconds([],2,[],d),0)
        self.assertEqual(simulate_department_wait_seconds([{"department":"CCS"}],1,[{"residual_seconds":30}],d),120)
        self.assertIsNone(simulate_department_wait_seconds([],0,[],d))

    def test_serialized_pipeline_prediction_contract(self):
        class Constant:
            def predict(self, rows): return [2.0]*len(rows)
        self.assertEqual(predict_service_time("CCS","10:30",0,model=Constant()),120)

    def test_nonnegative_prediction_clamped(self):
        class Negative:
            def predict(self, rows): return [-1.0]*len(rows)
        self.assertEqual(predict_service_time("CCS",model=Negative()),0)

    def test_zero_windows_unavailable(self):
        self.assertIsNone(predict_waiting_time([],0))

    def test_saved_model_serialization_and_loading(self):
        path=Path(__file__).resolve().parents[1]/"models"/"ets_model.joblib"
        self.assertTrue(path.exists(), "run src/train_models.py before model artifact test")
        artifact=load_model(path)
        self.assertEqual(artifact["version"],"1.0.0")
        self.assertIn("department",artifact["features"])


if __name__ == "__main__": unittest.main()
