# Lab 04 - A2
# Unit tests for the modular functions of Lab 03 (Lab3.py, my own code) and
# Lab 04 (Lab4.py, AI generated code). Both versions are tested by the same
# test bodies, so a test failing on one version and passing on the other
# immediately shows which implementation is wrong.
#
# GenAI tool used to generate these test cases: Claude (Anthropic) - Claude Code CLI, model Opus 5.
#
# Run with:  python -m unittest test_lab.py -v

import unittest
import numpy as np
import pandas as pd
import scipy.spatial.distance

import Lab3
import Lab4

# The two labs implement the same maths under different function names.
# Each pair is (version label, Lab3 function, Lab4 function).
LABEL_ENCODE = [("Lab3", Lab3.labelencoding), ("Lab4", Lab4.label_encode)]
ONE_HOT = [("Lab3", Lab3.onehot), ("Lab4", Lab4.one_hot_encode)]
ENCODE_DATASET = [("Lab3", Lab3.encodedataset), ("Lab4", Lab4.encode_dataset)]
MINKOWSKI = [("Lab3", Lab3.minkowsi), ("Lab4", Lab4.minkowski_distance)]
MINKOWSKI_RANGE = [("Lab3", Lab3.distance), ("Lab4", Lab4.minkowski_range)]
COMPARE_SCIPY = [("Lab3", Lab3.compare), ("Lab4", Lab4.compare_with_scipy)]
DOT = [("Lab3", Lab3.dotproduct), ("Lab4", Lab4.dot_product)]
NORM = [("Lab3", Lab3.norm), ("Lab4", Lab4.euclidean_norm)]
MEAN = [("Lab3", Lab3.mean), ("Lab4", Lab4.mean)]
VARIANCE = [("Lab3", Lab3.variance), ("Lab4", Lab4.variance)]
STD = [("Lab3", Lab3.std), ("Lab4", Lab4.std)]
MATRIX_STATS = [("Lab3", Lab3.matrixstats), ("Lab4", Lab4.matrix_stats)]
COMPARE_STATS = [("Lab3", Lab3.comparestats), ("Lab4", Lab4.compare_stats)]
HISTOGRAM = [("Lab3", Lab3.histdata), ("Lab4", Lab4.histogram_data)]


def small_dataframe():
    """Tiny stand-in for marketing_campaign so the tests need no Excel file."""
    return pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Education": ["PhD", "Basic", "Graduation", "PhD"],
        "Marital_Status": ["Single", "Married", "Single", "Widow"],
        "Dt_Customer": ["04-09-2012", "08-03-2014", "21-08-2013", "10-02-2014"],
        "Income": [58138.0, 46344.0, 71613.0, 26646.0],
    })


# A2 - label encoding
class TestLabelEncoding(unittest.TestCase):
    def test_categories_get_indices_in_sorted_order(self):
        values = ["PhD", "Basic", "Graduation", "PhD"]
        for version, encode in LABEL_ENCODE:
            with self.subTest(version):
                codes, mapping = encode(values)
                self.assertEqual(mapping, {"Basic": 0, "Graduation": 1, "PhD": 2})
                self.assertEqual(codes, [2, 0, 1, 2])

    def test_same_value_always_gets_same_code(self):
        for version, encode in LABEL_ENCODE:
            with self.subTest(version):
                codes, _ = encode(["a", "b", "a", "b", "a"])
                self.assertEqual(codes[0], codes[2])
                self.assertEqual(codes[1], codes[3])

    def test_single_category(self):
        for version, encode in LABEL_ENCODE:
            with self.subTest(version):
                codes, mapping = encode(["only", "only"])
                self.assertEqual(codes, [0, 0])
                self.assertEqual(mapping, {"only": 0})


# A2 - one hot encoding
class TestOneHotEncoding(unittest.TestCase):
    def test_shape_and_contents(self):
        values = ["Single", "Married", "Single"]
        for version, encode in ONE_HOT:
            with self.subTest(version):
                matrix, categories = encode(values)
                self.assertEqual(categories, ["Married", "Single"])
                self.assertEqual(matrix, [[0, 1], [1, 0], [0, 1]])

    def test_every_row_has_exactly_one_hot_bit(self):
        for version, encode in ONE_HOT:
            with self.subTest(version):
                matrix, categories = encode(["x", "y", "z", "y"])
                self.assertEqual(len(matrix), 4)
                for row in matrix:
                    self.assertEqual(len(row), len(categories))
                    self.assertEqual(sum(row), 1)


# A3 - dataset encoding / dimensionality
class TestEncodeDataset(unittest.TestCase):
    def test_categorical_columns_are_replaced(self):
        for version, encode in ENCODE_DATASET:
            with self.subTest(version):
                encoded, edu_map, marital_cols = encode(small_dataframe())
                self.assertNotIn("Dt_Customer", encoded.columns)
                self.assertNotIn("Marital_Status", encoded.columns)
                self.assertIn("Education", encoded.columns)
                for c in marital_cols:
                    self.assertIn("Marital_" + c, encoded.columns)

    def test_dimensionality_after_encoding(self):
        # 5 columns in, minus Dt_Customer and Marital_Status, plus 3 marital dummies
        for version, encode in ENCODE_DATASET:
            with self.subTest(version):
                encoded, edu_map, marital_cols = encode(small_dataframe())
                self.assertEqual(len(marital_cols), 3)
                self.assertEqual(encoded.shape, (4, 6))
                self.assertEqual(edu_map, {"Basic": 0, "Graduation": 1, "PhD": 2})

    def test_all_encoded_columns_are_numeric(self):
        for version, encode in ENCODE_DATASET:
            with self.subTest(version):
                encoded, _, _ = encode(small_dataframe())
                for column in encoded.columns:
                    self.assertTrue(pd.api.types.is_numeric_dtype(encoded[column]), column)


# A4 - Minkowski distance
class TestMinkowski(unittest.TestCase):
    def test_p1_is_manhattan(self):
        for version, minkowski in MINKOWSKI:
            with self.subTest(version):
                self.assertAlmostEqual(minkowski([1, 2, 3], [4, 6, 8], 1), 12.0)

    def test_p2_is_euclidean(self):
        for version, minkowski in MINKOWSKI:
            with self.subTest(version):
                self.assertAlmostEqual(minkowski([0, 0], [3, 4], 2), 5.0)

    def test_distance_to_itself_is_zero(self):
        for version, minkowski in MINKOWSKI:
            with self.subTest(version):
                for p in range(1, 6):
                    self.assertAlmostEqual(minkowski([2, 5, 9], [2, 5, 9], p), 0.0)

    def test_symmetry(self):
        a, b = [1, 7, 3], [4, 2, 8]
        for version, minkowski in MINKOWSKI:
            with self.subTest(version):
                for p in range(1, 6):
                    self.assertAlmostEqual(minkowski(a, b, p), minkowski(b, a, p))

    def test_negative_components_use_absolute_difference(self):
        for version, minkowski in MINKOWSKI:
            with self.subTest(version):
                self.assertAlmostEqual(minkowski([-1, -2], [1, 2], 1), 6.0)

    def test_distance_decreases_as_p_grows(self):
        a, b = [1, 2, 3], [4, 6, 8]
        for version, minkowski in MINKOWSKI:
            with self.subTest(version):
                values = [minkowski(a, b, p) for p in range(1, 11)]
                for earlier, later in zip(values, values[1:]):
                    self.assertGreaterEqual(earlier, later)


# A5 - distance over p = 1..10
class TestMinkowskiRange(unittest.TestCase):
    def test_returns_one_distance_per_p(self):
        for version, distance_range in MINKOWSKI_RANGE:
            with self.subTest(version):
                p_values, distances = distance_range([1, 2, 3], [4, 6, 8], 10)
                self.assertEqual(p_values, list(range(1, 11)))
                self.assertEqual(len(distances), 10)
                self.assertAlmostEqual(distances[0], 12.0)


# A6 - agreement with scipy
class TestAgainstScipy(unittest.TestCase):
    def test_matches_scipy_for_p_1_to_10(self):
        a, b = [1.0, 2.0, 3.0, 9.0], [4.0, 6.0, 8.0, 1.0]
        for version, compare in COMPARE_SCIPY:
            with self.subTest(version):
                rows = compare(a, b, 10)
                self.assertEqual(len(rows), 10)
                for p, mine, package, diff in rows:
                    self.assertAlmostEqual(mine, package, places=9)
                    self.assertLess(diff, 1e-9)

    def test_own_function_matches_scipy_directly(self):
        a, b = [2.0, 4.0, 6.0], [1.0, 9.0, 3.0]
        for version, minkowski in MINKOWSKI:
            with self.subTest(version):
                for p in range(1, 11):
                    self.assertAlmostEqual(
                        minkowski(a, b, p),
                        scipy.spatial.distance.minkowski(a, b, p), places=9)


# A7 - dot product and Euclidean norm
class TestVectorOperations(unittest.TestCase):
    def test_dot_product_matches_numpy(self):
        a, b = [1, 2, 3], [4, 5, 6]
        for version, dot in DOT:
            with self.subTest(version):
                self.assertAlmostEqual(dot(a, b), float(np.dot(a, b)))
                self.assertAlmostEqual(dot(a, b), 32.0)

    def test_dot_product_of_orthogonal_vectors_is_zero(self):
        for version, dot in DOT:
            with self.subTest(version):
                self.assertAlmostEqual(dot([1, 0], [0, 1]), 0.0)

    def test_norm_matches_numpy(self):
        vector = [3, 4, 12]
        for version, norm in NORM:
            with self.subTest(version):
                self.assertAlmostEqual(norm(vector), float(np.linalg.norm(vector)))
                self.assertAlmostEqual(norm([3, 4]), 5.0)

    def test_norm_of_zero_vector(self):
        for version, norm in NORM:
            with self.subTest(version):
                self.assertAlmostEqual(norm([0, 0, 0]), 0.0)


# A8 - mean, variance, standard deviation
class TestStatistics(unittest.TestCase):
    def test_mean(self):
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        for version, fn in MEAN:
            with self.subTest(version):
                self.assertAlmostEqual(fn(data), 5.0)
                self.assertAlmostEqual(fn(data), float(np.mean(data)))

    def test_variance_is_population_variance(self):
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        for version, fn in VARIANCE:
            with self.subTest(version):
                self.assertAlmostEqual(fn(data), 4.0)
                self.assertAlmostEqual(fn(data), float(np.var(data)))

    def test_std_is_sqrt_of_variance(self):
        data = [2, 4, 4, 4, 5, 5, 7, 9]
        for version, fn in STD:
            with self.subTest(version):
                self.assertAlmostEqual(fn(data), 2.0)
                self.assertAlmostEqual(fn(data), float(np.std(data)))

    def test_constant_data_has_zero_spread(self):
        for version, fn in VARIANCE:
            with self.subTest(version):
                self.assertAlmostEqual(fn([7, 7, 7, 7]), 0.0)

    def test_matrix_stats_are_column_wise(self):
        matrix = [[1, 10], [2, 20], [3, 30]]
        for version, fn in MATRIX_STATS:
            with self.subTest(version):
                means, variances, stds = fn(matrix)
                self.assertEqual(len(means), 2)
                np.testing.assert_allclose(means, np.mean(matrix, axis=0))
                np.testing.assert_allclose(variances, np.var(matrix, axis=0))
                np.testing.assert_allclose(stds, np.std(matrix, axis=0))


# A9 - own statistics vs numpy
class TestCompareStats(unittest.TestCase):
    def test_one_row_per_feature_with_negligible_difference(self):
        matrix = [[1.0, 10.0, -5.0], [2.0, 20.0, 0.0], [3.0, 30.0, 5.0]]
        for version, fn in COMPARE_STATS:
            with self.subTest(version):
                rows = fn(matrix)
                self.assertEqual(len(rows), 3)
                for row in rows:
                    self.assertEqual(len(row), 6)
                    self.assertLess(row[2], 1e-9)   # mean difference
                    self.assertLess(row[5], 1e-9)   # std difference


# A10 - histogram
class TestHistogram(unittest.TestCase):
    def test_counts_cover_every_observation(self):
        feature = [1, 2, 2, 3, 5, 8, 13, 21, 34, 55]
        for version, fn in HISTOGRAM:
            with self.subTest(version):
                counts, edges, m, v = fn(feature, 10)
                self.assertEqual(len(counts), 10)
                self.assertEqual(len(edges), 11)
                self.assertEqual(int(sum(counts)), len(feature))
                self.assertAlmostEqual(m, float(np.mean(feature)))
                self.assertAlmostEqual(v, float(np.var(feature)))

    def test_edges_span_the_data_range(self):
        feature = [0.0, 5.0, 10.0]
        for version, fn in HISTOGRAM:
            with self.subTest(version):
                counts, edges, _, _ = fn(feature, 5)
                self.assertAlmostEqual(edges[0], min(feature))
                self.assertAlmostEqual(edges[-1], max(feature))


# A11 - k-means (Lab4 only; Lab3.py has no k-means implementation yet)
class TestKMeans(unittest.TestCase):
    def setUp(self):
        # two well separated blobs, three points each
        self.matrix = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                       [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]]

    def test_assign_clusters_picks_the_nearest_centroid(self):
        centroids = [[0.0, 0.0], [10.0, 10.0]]
        labels = Lab4.assign_clusters(self.matrix, centroids)
        self.assertEqual(labels, [0, 0, 0, 1, 1, 1])

    def test_update_centroids_is_the_cluster_mean(self):
        labels = [0, 0, 0, 1, 1, 1]
        centroids = Lab4.update_centroids(self.matrix, labels, 2)
        np.testing.assert_allclose(centroids[0], np.mean(self.matrix[:3], axis=0))
        np.testing.assert_allclose(centroids[1], np.mean(self.matrix[3:], axis=0))

    def test_update_centroids_survives_an_empty_cluster(self):
        labels = [0, 0, 0, 0, 0, 0]          # cluster 1 gets nothing
        centroids = Lab4.update_centroids(self.matrix, labels, 2)
        self.assertEqual(len(centroids), 2)
        self.assertEqual(len(centroids[1]), 2)

    def test_sse_is_zero_when_points_sit_on_their_centroid(self):
        centroids = [[0.0, 0.0], [10.0, 10.0]]
        labels = [0, 1]
        self.assertAlmostEqual(Lab4.sse([[0.0, 0.0], [10.0, 10.0]], labels, centroids), 0.0)

    def test_recovers_two_obvious_clusters(self):
        centroids, labels, error, iterations = Lab4.kmeans(self.matrix, 2)
        self.assertEqual(len(centroids), 2)
        self.assertEqual(len(labels), len(self.matrix))
        # the first three points must share a label, and so must the last three
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertEqual(len(set(labels[3:])), 1)
        self.assertNotEqual(labels[0], labels[3])
        self.assertLess(error, 1.0)
        self.assertGreaterEqual(iterations, 1)

    def test_more_clusters_reduce_the_error(self):
        _, _, error_k2, _ = Lab4.kmeans(self.matrix, 2)
        _, _, error_k3, _ = Lab4.kmeans(self.matrix, 3)
        self.assertLessEqual(error_k3, error_k2)

    def test_same_seed_gives_the_same_result(self):
        first = Lab4.kmeans(self.matrix, 2, seed=42)
        second = Lab4.kmeans(self.matrix, 2, seed=42)
        self.assertEqual(first[1], second[1])
        np.testing.assert_allclose(first[0], second[0])

    def test_converges_before_the_iteration_cap(self):
        _, _, _, iterations = Lab4.kmeans(self.matrix, 2, max_iterations=100)
        self.assertLess(iterations, 100)


# Cross version check: the two implementations must agree with each other
class TestVersionsAgree(unittest.TestCase):
    def test_distance_matches_between_lab3_and_lab4(self):
        a, b = [1.0, 5.0, 3.0, 7.0], [4.0, 2.0, 8.0, 0.0]
        for p in range(1, 11):
            self.assertAlmostEqual(Lab3.minkowsi(a, b, p),
                                   Lab4.minkowski_distance(a, b, p), places=9)

    def test_statistics_match_between_lab3_and_lab4(self):
        matrix = [[1.0, 10.0], [2.0, 25.0], [3.0, 30.0], [4.0, 15.0]]
        lab3 = Lab3.matrixstats(matrix)
        lab4 = Lab4.matrix_stats(matrix)
        for mine, theirs in zip(lab3, lab4):
            np.testing.assert_allclose(mine, theirs)

    def test_encoding_matches_between_lab3_and_lab4(self):
        values = ["PhD", "Basic", "Graduation", "PhD", "Master"]
        self.assertEqual(Lab3.labelencoding(values), Lab4.label_encode(values))
        self.assertEqual(Lab3.onehot(values), Lab4.one_hot_encode(values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
