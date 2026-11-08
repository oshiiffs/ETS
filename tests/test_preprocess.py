import unittest
from src.preprocess import clean_rows
from src.predict import predict_wait_minutes
from src.evaluate import split_days
from datetime import datetime


class PipelineTests(unittest.TestCase):
    def test_wait_and_service(self):
        row={"date":"2026-01-01","ticket_number":"1","window":"W1","ticket_time":"2026-01-01 10:00:00","people_ahead":"0","windows_open":"1","called_time":"2026-01-01 10:03:00","finished_time":"2026-01-01 10:05:00"}
        out=clean_rows([row])[0]
        self.assertEqual(out["actual_wait_min"],"3.000000")
        self.assertEqual(out["actual_service_min"],"2.000000")

    def test_midnight_rollover(self):
        row={"date":"2026-01-01","ticket_number":"1","window":"W1","ticket_time":"2026-01-01 23:58:00","people_ahead":"0","windows_open":"1","called_time":"2026-01-01 00:03:00","finished_time":"2026-01-01 00:05:00"}
        out=clean_rows([row])[0]
        self.assertEqual(out["actual_wait_min"],"5.000000")
        self.assertIn("crossed_midnight",out["validation_note"])

    def test_predict_idle_and_reject_unknown_closed(self):
        now=datetime(2026,1,1)
        self.assertEqual(predict_wait_minutes(0,1,[],0,2,now),0)
        with self.assertRaises(ValueError): predict_wait_minutes(1,0,[],0,2,now)
        with self.assertRaises(ValueError): predict_wait_minutes(1,1,[],15,2,now)

    def test_split_keeps_complete_days(self):
        rows=[{"date":d} for d in ("2026-01-01","2026-01-01","2026-01-02","2026-01-03")]
        train,test,_,_=split_days(rows)
        self.assertEqual({r["date"] for r in train},{"2026-01-01","2026-01-02"})
        self.assertEqual({r["date"] for r in test},{"2026-01-03"})


if __name__=="__main__": unittest.main()
