import unittest

from director.llm import LLMError, parse_json


class ParseJsonTests(unittest.TestCase):
    def test_extracts_unicode_and_empty_values_from_fenced_reply(self):
        reply = "```json\n{\"message\": \"雪\", \"empty\": \"\", \"items\": []}\n```"

        self.assertEqual(
            parse_json(reply),
            {"message": "雪", "empty": "", "items": []},
        )

    def test_rejects_empty_and_non_object_replies(self):
        for reply in ("", "   ", None, "[1, 2, 3]"):
            with self.subTest(reply=reply):
                with self.assertRaisesRegex(LLMError, "no JSON object in reply"):
                    parse_json(reply)


if __name__ == "__main__":
    unittest.main()
