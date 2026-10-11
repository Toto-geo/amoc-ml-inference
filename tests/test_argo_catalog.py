"""Argo QC and real-catalog selection must remain valid as the live GDAC changes."""

import unittest

import pandas as pd

from data_preparation.dataset_construction.derived.argo.build_argo_qc import build as qc_build
from data_preparation.dataset_construction.derived.argo.build_argo_geometry import real_catalog


class ArgoCatalogTest(unittest.TestCase):
    def test_qc_and_real_catalog_do_not_require_historical_counts(self):
        rows = pd.DataFrame({
            "wmo": [100001, 100002, 100003],
            "cyc": [1, 2, 3],
            "date": ["2018-01-16", "2019-02-16", "2019-03-16"],
            "lat": [-35.0] * 3,
            "lon": [-20.0] * 3,
            "http_status": [200, 200, 404],
            "error": [None, None, "not found"],
            "POSITION_QC": [1] * 3,
            "JULD_QC": [1] * 3,
            "DATA_MODE": ["D"] * 3,
            "n_levels_0_2000": [50] * 3,
            "n_good_pres_0_2000": [50] * 3,
            "n_good_temp_0_2000": [50] * 3,
            "n_good_psal_0_2000": [50] * 3,
            "frac_good_temp_0_2000": [1.0] * 3,
            "frac_good_psal_0_2000": [1.0] * 3,
            "max_pres_good": [2000.0] * 3,
        })
        qc = qc_build(rows)
        self.assertEqual(len(qc), 2)
        self.assertEqual(real_catalog(qc).profile_id.tolist(), ["100001_0001", "100002_0002"])


if __name__ == "__main__":
    unittest.main()
