import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.rules_loader import (
    DEFAULT_RULES_PATH,
    expand_abbreviation,
    get_icd10,
    get_mandatory_fields,
    get_risk_tier,
    get_rules_sha256,
    get_weight,
    is_hard_guardrail,
    load_rules,
)


class RulesLoaderTests(unittest.TestCase):
    def test_load_rules_returns_expected_structure(self):
        rules = load_rules(DEFAULT_RULES_PATH)
        self.assertIn("mandatory_clinical_fields", rules)
        self.assertIn("normalization_standards", rules)
        self.assertIn("risk_scoring_matrix", rules)

    def test_get_mandatory_fields_for_clinical(self):
        fields = get_mandatory_fields("clinical")
        self.assertIn("patient_id", fields)
        self.assertIn("discharge_diagnosis", fields)
        self.assertIn("discharge_instructions", fields)

    def test_get_mandatory_fields_for_prescription(self):
        fields = get_mandatory_fields("prescription")
        self.assertIn("medicine_name", fields)
        self.assertIn("total_quantity", fields)

    def test_get_weight_for_known_key(self):
        self.assertEqual(get_weight("allergy_contradiction"), 8)
        self.assertEqual(get_weight("missing_address"), 1)

    def test_get_risk_tier(self):
        self.assertEqual(get_risk_tier(2), "low")
        self.assertEqual(get_risk_tier(5), "medium")
        self.assertEqual(get_risk_tier(9), "high")

    def test_is_hard_guardrail(self):
        self.assertTrue(is_hard_guardrail("allergy_contradiction"))
        self.assertFalse(is_hard_guardrail("missing_address"))

    def test_expand_abbreviation(self):
        self.assertEqual(expand_abbreviation("HTN"), "Hypertension")
        self.assertEqual(expand_abbreviation("XYZ"), "XYZ")

    def test_get_icd10(self):
        self.assertEqual(get_icd10("Type 2 Diabetes Mellitus"), "E11.9")
        self.assertIsNone(get_icd10("Unknown Disease"))

    def test_rules_file_exists(self):
        self.assertTrue(Path(DEFAULT_RULES_PATH).exists())

    def test_rules_sha256_is_stable(self):
        first = get_rules_sha256(DEFAULT_RULES_PATH)
        second = get_rules_sha256(DEFAULT_RULES_PATH)
        self.assertEqual(first, second)
        self.assertTrue(len(first) == 64)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RulesLoaderTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"\nPassed {result.testsRun} test case(s)")
