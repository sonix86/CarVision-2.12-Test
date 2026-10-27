"""Public diagnostic wording must preserve the actual acceptance decision."""
import unittest
from carvision.vehicle_model import model_status_text, vehicle_name


class ModelStatusChecks(unittest.TestCase):
    def test_leading_dolphin_is_a_hypothesis_below_threshold(self):
        row=dict(make_first=True,brand='BYD',model='unknown',model_threshold=.7,
                 model_alternatives=[dict(brand='BYD',model='Dolphin',confidence=.34)])
        self.assertIn('Dolphin',model_status_text(row))
        self.assertIn('34%',model_status_text(row))
        self.assertIn('70%',model_status_text(row))
        self.assertEqual(vehicle_name(row),'BYD · модель не определена')

    def test_uncertain_make_does_not_claim_missing_model_training(self):
        row=dict(make_first=True,brand='unknown',model_reason='make not accepted')
        self.assertIn('подтвердить марку',model_status_text(row))
        self.assertNotIn('нет доступных',model_status_text(row))

    def test_cross_make_guess_is_not_presented_as_same_make_family(self):
        row=dict(brand='BYD',model='unknown',model_reason='unsupported',model_alternatives=[dict(brand='Chevrolet',model='Captiva',confidence=.9)])
        self.assertNotIn('Captiva',model_status_text(row))

    def test_crop_limitation_is_distinct_from_missing_training(self):
        row=dict(brand='BYD',model='unknown',model_limited=True,model_alternatives=[dict(brand='BYD',model='Dolphin',confidence=.9)])
        self.assertIn('видимых деталей',model_status_text(row))

    def test_accepted_model_has_no_rejection_warning(self):
        self.assertEqual(model_status_text(dict(brand='BYD',model='Dolphin')),'')
