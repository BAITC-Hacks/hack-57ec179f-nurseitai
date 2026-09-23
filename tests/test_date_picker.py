import unittest
from datetime import date

from date_picker import decode_date


class DatePickerTests(unittest.TestCase):
    def decode(self, value, error=None):
        return decode_date({"value": value, "error": error}, date(2024, 1, 1), date(2026, 12, 31))

    def test_real_dates_and_leap_year(self):
        self.assertEqual(self.decode("2026-10-10"), date(2026, 10, 10))
        self.assertEqual(self.decode("2024-02-29"), date(2024, 2, 29))
        for value in ("2026-02-29", "2026-11-31", "2026-13-01", "", None,
                      "10.10.2026", "2026-1-1", "2026-10-10T00:00:00"):
            with self.subTest(value=value):
                self.assertIsNone(self.decode(value))

    def test_bounds_and_invalid_drafts(self):
        for value in ("2023-12-31", "2027-01-01"):
            self.assertIsNone(self.decode(value))
        self.assertEqual(self.decode("2024-01-01"), date(2024, 1, 1))
        self.assertEqual(self.decode("2026-12-31"), date(2026, 12, 31))
        self.assertIsNone(self.decode("2026-10-10", "Ошибка"))
        self.assertIsNone(decode_date("2026-10-10", date(2024, 1, 1), date(2026, 12, 31)))
