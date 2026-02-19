from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm


def initialize_api_client(api_key: str, provider: str = "fireworks") -> OpenAI:
    """Initializes and validates the API client.

    Args:
        api_key: API key for the provider
        provider: "fireworks" or "openai"
    """
    if provider == "openai":
        return OpenAI(api_key=api_key)

    return OpenAI(
        base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1/"),
        api_key=api_key,
    )


def repair_json_errors(json_str: str) -> str:
    """Repair common JSON errors from LLM output."""
    pattern = r'(\{"lang":\s*"[^"]+",\s+)"([^"]+)"'

    def fix_text_key(match):
        prefix = match.group(1)
        content = match.group(2)
        if content not in ['text', 'requirement', 'lang', 'item_no', 'item_name', 'rules', 'type', 'item']:
            return f'{prefix}"text": "{content}"'
        return match.group(0)

    json_str = re.sub(pattern, fix_text_key, json_str)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str


def parse_section_with_llm(
    section: dict,
    api_client: OpenAI,
    parser_model: str,
) -> Optional[dict]:
    """Parse a section using LLM with strict instructions."""
    legal_term_json = json.dumps(section, indent=2, ensure_ascii=False)
    prompt = f"""
              You are an expert JSON generator specializing in legal packaging requirements parser.
              Extract rules from legal terms into a structured, machine-readable JSON format.

              **CRITICAL RULES - READ CAREFULLY:**

              1. **Extract Requirements EXACTLY as Stated - DO NOT Over-Interpret**
                  - If legal term says: "The statement must be available" → requirement: ["text_available = true"]
                  - If legal term says: "Text height must be >= 2mm" → requirement: ["text_height >= 2 mm"]
                  - DO NOT infer semantic meaning from text content
                  - DO NOT add requirements based on what the text says
                  - ONLY extract what is EXPLICITLY stated in the "requirement" field
                  - do not put literal text

              2. **Text Extraction Rules**
                  - ALWAYS extract the SPECIFIC TEXT that needs validation from "remarks" field AND "language reference" field
                  - For addresses: Extract the complete address from remarks
                  - For warnings: Extract the complete warning text from language reference
                  - For multi-line text with \\n: Split into separate text entries
                  - DO NOT extract the requirement description itself
                  - PRESERVE all non-Latin characters (Arabic, Russian, Chinese, etc.) exactly as-is
                  - The "text" field should contain the ACTUAL TEXT from remarks/language reference
                  - The "requirement" field should contain machine-executable validation rules like:
                    "text_height >= 1.5 mm"
                    "position = front_panel"
                    "applies_to_country = ['Brazil', 'France']"
                    "text_available = true"
                  - CRITICAL: If "remarks" or "language reference" contains actual text (not instructions), extract it to "text" field!

              2a. **SPECIAL CASE: Trademark Symbols**
                  - If remarks mention "TM", "trademark symbol", "®", or "™" → Extract the ACTUAL SYMBOLS: "™" and "®"
                  - DO NOT extract literal text like "TM on Brand Name"
                  - Example: remarks="TM on Hot Wheels Brand Name" → text: [{{"lang": "English", "text": "™"}}, {{"lang": "English", "text": "®"}}]

              3. **Parent-Child Rule Detection**
                  - If a rule says "Follow steps X-Y" or "See items X-Y" or "refer to rule X":
                    * Mark as "is_reference_only": true
                    * Extract ALL referenced rule IDs: "refer_to_rules = ['X', 'Y', 'Z']"
                    * Example: "Follow steps 1.1 and 2.6" → requirement: ["refer_to_rules = ['1.1', '2.6']"]

                  - If rule number has decimal children (e.g., 3.1 has 3.1.1):
                    * Parent WITH text in remarks: Extract text from remarks + mark as parent
                      Example: Rule 3.1 remarks="Conforms to ASTM F963" → Extract this text + "is_parent_rule": true
                    * Parent WITHOUT text in remarks: Empty text array + mark as parent
                      Example: Rule 2.6 remarks="(see child rules)" → text: [] + "is_parent_rule": true
                    * Children:
                      - If parent HAS text: Leave child text EMPTY (will inherit from parent)
                      - If parent has NO text: Extract child-specific text from remarks

              4. **Conditional Requirements**
                  - If multiple statements with different conditions: Create SEPARATE rules

              **Input Legal Term:**
              {legal_term_json}

              **Output Format:**
              {{
                "type": "<country/region>",
                "item": [
                  {{
                    "item_no": "<number>",
                    "item_name": "<name>",
                    "rules": [
                      {{
                        "text": [
                          {{"lang": "<language>", "text": "<actual text from remarks>"}}
                        ],
                        "requirement": ["<requirement exactly as stated>"]
                      }}
                    ],
                    "is_parent_rule": true,
                    "child_rules": ["X.Y.Z"],
                    "parent_rule": "X.Y",
                    "is_reference_only": true,
                    "refer_to_rules": ["A", "B"]
                  }}
                ]
              }}

              **Return ONLY valid JSON, no explanations.**
              """

    try:
        response = api_client.chat.completions.create(
            model=parser_model,
            messages=[
                {"role": "system", "content": "You are a precise legal requirements parser. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4000,
            temperature=0.0,
        )
        result_text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', result_text or "", re.DOTALL)
        if not json_match:
            return None

        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            repaired_json = repair_json_errors(json_str)
            try:
                return json.loads(repaired_json)
            except json.JSONDecodeError:
                return None

    except Exception:
        return None


def clean_instructional_text(rules: dict, original_section: dict) -> dict:
    """Remove instructional text that shouldn't be in text field."""
    original_reqs = {}
    original_remarks = {}
    for item in original_section.get('item', []):
        item_no = str(item.get('no', ''))
        requirement = str(item.get('requirement', '')).lower()
        remarks = item.get('remarks', [])
        original_reqs[item_no] = requirement
        original_remarks[item_no] = remarks

    for item in rules.get('item', []):
        item_no = str(item.get('item_no', ''))
        original_req = original_reqs.get(item_no, '')
        original_remark = original_remarks.get(item_no, [])

        for rule in item.get('rules', []):
            text_field = rule.get('text', [])
            cleaned_texts = []

            for text_entry in text_field:
                text = text_entry.get('text', '').lower()
                if not text.strip():
                    continue

                requirement_indicators = ['must be', 'must have', 'shall be', 'should be', 'required', 'please refer']
                has_requirement_language = any(indicator in text for indicator in requirement_indicators)

                text_in_remarks = False
                if isinstance(original_remark, list):
                    for remark in original_remark:
                        remark_lower = str(remark).lower()
                        if len(text) > 10:
                            text_words = set(text.split())
                            remark_words = set(remark_lower.split())
                            overlap = len(text_words & remark_words) / len(text_words) if text_words else 0
                            if overlap > 0.5:
                                text_in_remarks = True
                                break
                        else:
                            if text in remark_lower:
                                text_in_remarks = True
                                break

                category_patterns = [
                    r'^[a-z\s]+statement$',
                    r'^[a-z\s]+statements$',
                    r'^directions for',
                    r'^instructions for',
                    r'^package copy for',
                ]
                is_category_label = any(re.match(pattern, text) for pattern in category_patterns)

                if has_requirement_language and not text_in_remarks:
                    continue
                if is_category_label:
                    continue

                cleaned_texts.append(text_entry)

            rule['text'] = cleaned_texts

    return rules


def detect_parent_child_relationships(rules: dict) -> dict:
    """Detect parent-child relationships based on rule numbering."""
    items = rules.get('item', [])
    rule_index = {}
    for item in items:
        item_no = str(item.get('item_no', ''))
        rule_index[item_no] = item

    for item_no, item in rule_index.items():
        child_rules = []
        for other_no in rule_index.keys():
            if other_no.startswith(item_no + '.') and other_no != item_no:
                remainder = other_no[len(item_no) + 1:]
                if '.' not in remainder:
                    child_rules.append(other_no)

        if child_rules:
            item['is_parent_rule'] = True
            item['child_rules'] = sorted(child_rules)

            for rule in item.get('rules', []):
                if rule.get('text'):
                    rule['text'] = []

            parent_reqs = set()
            for rule in item.get('rules', []):
                parent_reqs.update(rule.get('requirement', []))

            for child_no in child_rules:
                child_item = rule_index.get(child_no)
                if not child_item:
                    continue

                child_item['parent_rule'] = item_no
                for rule in child_item.get('rules', []):
                    child_reqs = rule.get('requirement', [])
                    unique_reqs = [req for req in child_reqs if req not in parent_reqs]
                    rule['requirement'] = unique_reqs

    rules['item'] = list(rule_index.values())
    return rules


def simplify_overspecified_requirements(rules: dict, original_section: dict) -> dict:
    """Simplify requirements that are over-specified by LLM."""
    original_reqs = {}
    for item in original_section.get('item', []):
        item_no = str(item.get('no', ''))
        requirement = str(item.get('requirement', '')).lower()
        original_reqs[item_no] = requirement

    for item in rules.get('item', []):
        item_no = str(item.get('item_no', ''))
        original_req = original_reqs.get(item_no, '')

        for rule in item.get('rules', []):
            requirements = rule.get('requirement', [])
            if original_req:
                simple_availability_indicators = ['must be available', 'statement must be', 'must be provided', 'must be located']
                is_simple_availability = any(indicator in original_req for indicator in simple_availability_indicators)
                has_complex_operators = any(op in str(requirements) for op in ['==', '>=', '<=', '!=', '&&', '||'])
                if is_simple_availability and has_complex_operators and len(requirements) > 1:
                    rule['requirement'] = ['text_available = true']

    return rules


def split_multiline_text(rules: dict) -> dict:
    """Split multi-line text (with \\n or \\n\\n) into separate entries."""
    for item in rules.get('item', []):
        for rule in item.get('rules', []):
            text_field = rule.get('text', [])
            fixed_texts = []
            for text_entry in text_field:
                text = text_entry.get('text', '')
                if '\n' in text:
                    parts = [p.strip() for p in text.split('\n') if p.strip()]
                    if len(parts) > 1:
                        for part in parts:
                            fixed_texts.append({'lang': text_entry.get('lang'), 'text': part})
                    else:
                        fixed_texts.append(text_entry)
                else:
                    fixed_texts.append(text_entry)

            rule['text'] = fixed_texts

    return rules


def clean_metadata_requirements(rules: dict) -> dict:
    """Remove metadata from requirements (e.g., country names, type names)."""
    rule_type = rules.get('type', '')
    for item in rules.get('item', []):
        for rule in item.get('rules', []):
            requirements = rule.get('requirement', [])
            cleaned = []
            for req in requirements:
                req_lower = str(req).lower()
                if req_lower.startswith('applies_to_'):
                    continue
                if rule_type and rule_type.lower() in req_lower:
                    if '=' in req_lower and not any(op in req_lower for op in ['>=', '<=', '!=']):
                        continue
                cleaned.append(req)
            rule['requirement'] = [req for req in cleaned if not str(req).lower().startswith('applies_to_')]

    return rules


def detect_reference_rules(rules: dict, original_section: dict) -> dict:
    """Detect rules that are just references to other rules."""
    original_reqs = {}
    original_remarks = {}

    for item in original_section.get('item', []):
        item_no = str(item.get('no', ''))
        requirement = str(item.get('requirement', '')).lower()
        remarks = item.get('remarks', [])
        original_reqs[item_no] = requirement
        original_remarks[item_no] = ' '.join(str(r).lower() for r in remarks) if isinstance(remarks, list) else str(remarks).lower()

    for item in rules.get('item', []):
        item_no = str(item.get('item_no', ''))
        original_req = original_reqs.get(item_no, '')
        original_remark = original_remarks.get(item_no, '')
        combined_text = original_req + ' ' + original_remark

        reference_pattern = r'(follow|refer|see|reference)\s+(steps?|items?|points?|section)\s+([\d\.\s,-]+)'
        ref_match = re.search(reference_pattern, combined_text)
        if not ref_match:
            continue

        item['is_reference_only'] = True
        for rule in item.get('rules', []):
            rule['text'] = []
            ref_text = ref_match.group(3)
            range_match = re.search(r'([\d\.]+)\s*-\s*([\d\.]+)', ref_text)
            if range_match:
                rule['requirement'] = [f"refer_to_rules = ['{range_match.group(1)}' to '{range_match.group(2)}']"]
            else:
                numbers = re.findall(r'[\d\.]+', ref_text)
                if numbers:
                    rule['requirement'] = [f"refer_to_rules = {numbers}"]
                else:
                    rule['requirement'] = ['reference_only = true']

    return rules


def split_conditional_requirements(rules: dict, original_section: dict) -> dict:
    """Split rules with multiple conditional statements."""
    original_remarks = {}
    for item in original_section.get('item', []):
        item_no = str(item.get('no', ''))
        remarks = item.get('remarks', [])
        original_remarks[item_no] = remarks

    new_items = []
    for item in rules.get('item', []):
        item_no = str(item.get('item_no', ''))
        remarks = original_remarks.get(item_no, [])

        if len(remarks) > 1:
            for rule in item.get('rules', []):
                texts = rule.get('text', [])
                requirements = rule.get('requirement', [])
                has_conditional_logic = any(
                    any(keyword in str(req).lower() for keyword in ['applies', 'if', 'when', 'for', 'statement'])
                    for req in requirements
                )

                if len(texts) > 1 and len(requirements) > 1 and has_conditional_logic:
                    split_rules = []
                    for i, text_entry in enumerate(texts):
                        if i < len(requirements):
                            split_rules.append({'text': [text_entry], 'requirement': [requirements[i]]})
                    item['rules'] = split_rules

        new_items.append(item)

    rules['item'] = new_items
    return rules


def apply_comprehensive_postprocessing(parsed_rules: dict, original_section: dict) -> dict:
    parsed_rules = clean_instructional_text(parsed_rules, original_section)
    parsed_rules = detect_parent_child_relationships(parsed_rules)
    parsed_rules = simplify_overspecified_requirements(parsed_rules, original_section)
    parsed_rules = split_multiline_text(parsed_rules)
    parsed_rules = clean_metadata_requirements(parsed_rules)
    parsed_rules = detect_reference_rules(parsed_rules, original_section)
    parsed_rules = split_conditional_requirements(parsed_rules, original_section)
    return parsed_rules


def process_legal_terms(
    *,
    input_path: str,
    output_path: str,
    delay_between_calls: float,
    parser_api_key: str,
    parser_provider: str,
    parser_model: str,
) -> None:
    """Reads legal terms JSON, processes them using LLM and post-processing, saves normalized rules."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_normalized_rules = []
    api_client = initialize_api_client(parser_api_key, provider=parser_provider)

    for section in tqdm(data, desc="Processing Legal Sections"):
        section_type = section.get("type", "Unknown")
        items = section.get("item", [])
        if not items:
            continue

        parsed_section = parse_section_with_llm(section, api_client, parser_model=parser_model)
        if parsed_section:
            parsed_section = apply_comprehensive_postprocessing(parsed_section, section)
            all_normalized_rules.append(parsed_section)

        time.sleep(delay_between_calls)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_normalized_rules, f, ensure_ascii=False, indent=2)
