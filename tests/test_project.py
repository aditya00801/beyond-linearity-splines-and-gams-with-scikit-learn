import unittest

import numpy as np
import pandas as pd
import model

from beyond_linearity import (
    age_grid,
    gam_model,
    one_se_rule,
    poly_model,
    split_wage,
)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "age": np.arange(20, 40),
                "year": np.tile([2003, 2004, 2005, 2006], 5),
                "education": ["High School", "College"] * 10,
                "wage": np.linspace(30, 100, 20),
            }
        )

    def test_split_keeps_all_rows(self):
        train, test = split_wage(self.frame, test_size=0.25, random_state=3)

        self.assertEqual(len(train), 15)
        self.assertEqual(len(test), 5)
        self.assertEqual(set(train.index) | set(test.index), set(self.frame.index))
        self.assertTrue(set(train.index).isdisjoint(test.index))

    def test_age_grid_has_requested_endpoints(self):
        grid = age_grid(lo=25, hi=35, n=3)

        self.assertEqual(grid["age"].tolist(), [25.0, 30.0, 35.0])

    def test_original_model_module_keeps_its_public_functions(self):
        self.assertIs(model.poly_model, poly_model)

    def test_one_se_rule_prefers_simpler_eligible_value(self):
        selected = one_se_rule([1, 2, 3], [10.0, 9.1, 9.0], [0.1, 0.6, 0.2])

        self.assertEqual(selected, 2)

    def test_polynomial_model_fits_and_predicts(self):
        X = self.frame[["age"]]
        model = poly_model(2).fit(X, self.frame["wage"])

        predictions = model.predict(X)
        self.assertEqual(predictions.shape, (len(self.frame),))
        self.assertTrue(np.isfinite(predictions).all())

    def test_gam_model_accepts_numeric_and_categorical_features(self):
        X = self.frame[["age", "year", "education"]]
        model = gam_model(age_knots=3, year_knots=3).fit(X, self.frame["wage"])

        predictions = model.predict(X)
        self.assertEqual(predictions.shape, (len(self.frame),))
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
