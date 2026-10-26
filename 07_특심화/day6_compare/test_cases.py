"""제공 테스트: 실습07의 구현 후 실행. 새 모델을 배포하거나 test 성능을 측정하지 않습니다.
실행: python -m unittest -v test_cases
입력 하나를 바꾼 뒤 예상 성공/실패를 적고 추가 테스트를 작성해도 됩니다.
"""
import unittest
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from course_io import load_data, check_features, read_table, snapshot
from preprocessing import make_preprocessor
from checks import before_model, prediction_contract
from evaluate import regression_metrics, classification_metrics

ROOT = Path(__file__).resolve().parent

class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        cls.d = load_data(ROOT/'data')
        cls.model = Pipeline([('pre', make_preprocessor()),
                              ('model', KNeighborsClassifier(n_neighbors=3))])
        cls.model.fit(cls.d.X_train, cls.d.yc_train)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def test_01_split_ids_are_disjoint(self):
        train = set(self.d.X_train.index)
        valid = set(self.d.X_valid.index)
        test = set(self.d.test_ids)
        self.assertEqual((len(train),len(valid),len(test)),(60,20,20))
        self.assertFalse(train & valid or train & test or valid & test)
        self.assertEqual(len(train | valid | test),100)

    def test_02_perfect_regression_metrics(self):
        metrics = regression_metrics([10,20,30],[10,20,30])
        self.assertAlmostEqual(metrics['MAE_ms'],0)
        self.assertAlmostEqual(metrics['RMSE_ms'],0)
        self.assertAlmostEqual(metrics['R2'],1)

    def test_03_positive_class_is_failure(self):
        metrics = classification_metrics([0,1,1],[0,0,1])
        self.assertEqual((metrics['TN'],metrics['FP'],metrics['FN'],metrics['TP']), (1,0,1,1))
        self.assertAlmostEqual(metrics['Recall'],0.5)

    def test_04_unknown_category_keeps_eight_columns(self):
        new = read_table(ROOT/'data/new_requests.csv').set_index('request_id')
        result = prediction_contract(self.model,new)
        self.assertEqual(result['n_features'],8)
        self.assertTrue(result['statistics_unchanged'])

    def test_05_missing_required_column_fails(self):
        with self.assertRaises(ValueError):
            prediction_contract(self.model,self.d.X_valid.drop(columns=['priority']))

    def test_06_too_many_neighbors_fails_before_fit(self):
        with self.assertRaises(ValueError):
            before_model(KNeighborsClassifier(n_neighbors=61),self.d.X_train,self.d.yc_train,
                         self.d.X_valid,self.d.yc_valid)

    def test_07_predict_preserves_training_statistics(self):
        before = snapshot(self.model['pre'])
        self.model.predict(self.d.X_valid)
        self.assertEqual(before,snapshot(self.model['pre']))

    def test_08_missing_new_numeric_values_use_train_rules(self):
        new = self.d.X_valid.iloc[:2].copy()
        new[['input_chars','allowed_tools']] = np.nan
        result = prediction_contract(self.model,new)
        self.assertEqual(result['n_rows'],2)
        self.assertTrue(result['statistics_unchanged'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
