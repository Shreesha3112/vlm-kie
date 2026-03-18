"""Tests for json_repair utilities."""

import pytest
from vlm_kie.utils.json_repair import extract_json, parse_llm_json


class TestExtractJson:
    def test_plain_json_object(self):
        text = '{"key": "value"}'
        assert extract_json(text) == '{"key": "value"}'

    def test_markdown_fence(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_markdown_fence_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_leading_trailing_text(self):
        text = 'Here is the result:\n{"key": "value"}\nDone.'
        result = extract_json(text)
        assert result == '{"key": "value"}'

    def test_nested_json(self):
        text = '{"a": {"b": 1}, "c": [1, 2]}'
        assert extract_json(text) == '{"a": {"b": 1}, "c": [1, 2]}'

    def test_array(self):
        text = '[{"item": 1}, {"item": 2}]'
        assert extract_json(text) == '[{"item": 1}, {"item": 2}]'


class TestParseLLMJson:
    def test_valid_json(self):
        result = parse_llm_json('{"invoice_number": "INV-001", "total": 100.0}')
        assert result["invoice_number"] == "INV-001"
        assert result["total"] == 100.0

    def test_markdown_wrapped(self):
        text = '```json\n{"invoice_number": "INV-001"}\n```'
        result = parse_llm_json(text)
        assert result["invoice_number"] == "INV-001"

    def test_trailing_comma(self):
        text = '{"a": 1, "b": 2,}'
        result = parse_llm_json(text)
        assert result is not None
        assert result["a"] == 1

    def test_null_values(self):
        text = '{"invoice_number": null, "total": null}'
        result = parse_llm_json(text)
        assert result["invoice_number"] is None

    def test_completely_invalid(self):
        result = parse_llm_json("I cannot extract this information.")
        assert result is None

    def test_nested_line_items(self):
        text = '{"line_items": [{"description": "Item A", "quantity": 2, "unit_price": 5.0, "total": 10.0}]}'
        result = parse_llm_json(text)
        assert len(result["line_items"]) == 1
        assert result["line_items"][0]["description"] == "Item A"
