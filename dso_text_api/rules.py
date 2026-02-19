from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import APIError, OpenAI, RateLimitError
from tqdm import tqdm


class SchemaAnalyzer:
    """Analyzes rule requirements to determine which validation tools are needed."""

    def __init__(self, api_client: OpenAI):
        self.api_client = api_client
        self.requirement_patterns = self._build_pattern_cache()

    def _build_pattern_cache(self) -> Dict[str, List[str]]:
        return {
            "measurement": ["height", "size", "mm", "cm", ">=", "<=", "dimension", "font_size"],
            "position": ["position", "panel", "location", "placement", "bottom", "front", "top", "side"],
            "metadata": ["country", "applies_to", "metadata", "property", "product_"],
            "text_presence": ["text", "available", "present", "language", "matches"],
        }

    def analyze_requirement(self, requirement: str, required_text: List[Dict]) -> Dict[str, Any]:
        req_lower = requirement.lower()
        tools_needed = set()
        validation_types = []

        for val_type, patterns in self.requirement_patterns.items():
            if any(pattern in req_lower for pattern in patterns):
                validation_types.append(val_type)

        if required_text:
            tools_needed.add("find_text_blocks")

        if "measurement" in validation_types:
            tools_needed.add("get_font_size")

        if "position" in validation_types:
            tools_needed.add("get_position")

        if "metadata" in validation_types:
            tools_needed.add("get_product_metadata")

        confidence = min(1.0, len(validation_types) * 0.3 + 0.4)

        return {
            "tools_needed": list(tools_needed),
            "validation_types": validation_types,
            "confidence": confidence,
            "reasoning": f"Detected {', '.join(validation_types)} validation based on keywords in requirement",
        }

    def learn_from_execution(self, requirement: str, plan: Dict, execution_result: Dict):
        if not execution_result.get("success", False):
            error_msg = execution_result.get("error", "")
            if "variable" in error_msg.lower() or "index" in error_msg.lower():
                print(f"[LEARN] Detected potential incomplete plan for: {requirement[:50]}...")


class PlanValidator:
    """Validates execution plans to ensure they check all required aspects."""

    def __init__(self, schema_analyzer: SchemaAnalyzer):
        self.schema_analyzer = schema_analyzer

    def validate_plan_completeness(self, requirement: str, required_text: List[Dict], plan: Dict) -> Tuple[bool, str]:
        if not plan or "plan" not in plan:
            return False, "Plan is empty or malformed"

        analysis = self.schema_analyzer.analyze_requirement(requirement, required_text)
        tools_needed = set(analysis["tools_needed"])

        tools_used = set()
        for step in plan["plan"]:
            if "tool" in step:
                tools_used.add(step["tool"])

        missing_tools = tools_needed - tools_used

        if missing_tools:
            return (
                False,
                f"Plan is incomplete. Missing tools: {', '.join(missing_tools)}. Requirement mentions {', '.join(analysis['validation_types'])} but plan doesn't validate all aspects.",
            )

        has_assertion = any(step.get("tool") == "assert" for step in plan["plan"])
        if not has_assertion:
            return False, "Plan has no assertion step to validate the condition"

        return True, "Plan is complete"


PROMPT_VERSION = "v3.0_meta_learning"

PROMPT_TEMPLATE = """
You are an expert validation planning agent with deep understanding of packaging compliance rules.

**YOUR TASK:** Create a precise JSON execution plan to verify a packaging rule against OCR data.

**AVAILABLE TOOLS:**
{tools_definition_string}

**CONTEXT:**
1. **OCR Data Summary:** {ocr_summary}
2. **Product Metadata:** {product_metadata_json}
3. **Rule to Verify:** "{condition_string}"
4. **Required Text (CRITICAL):** {required_text_json}
    - These are the EXACT texts that MUST appear on the packaging
    - Use these as search keywords in find_text_blocks()
    - Search for exact phrases first, then try partial matches if needed
    - Example keywords: {keywords_hint}

**PLANNING RULES:**
1. **Use Required Text as exact search keywords** - Don't use generic terms
2. **Match validation to requirement keywords:**
    - "height", "size", "mm" → use get_font_size()
    - "position", "panel", "location" → use get_position()
    - "country", "metadata" → use get_product_metadata()
3. **Safety checks:**
    - ALWAYS check list length before accessing: `len($blocks) > 0 and ...`
    - Use bracket notation for dicts: `$blocks[0]['id']` not `$blocks[0].id`
    - Never use literal objects in assertions
4. **Tool returns:**
    - get_position() → {{"ymax_relative": float, "ymin_relative": float}}
    - get_font_size() → float (mm)
    - find_text_blocks() → [{{"id": int, "text": str}}]

NOW GENERATE THE PLAN. Respond ONLY with valid JSON, no explanations or markdown.
"""


class RuleEngine:
    def __init__(
        self,
        rules_path: str,
        ocr_data: List[Dict],
        image_metadata: Dict,
        product_metadata: Dict,
        api_client: OpenAI,
        plan_cache_path: str,
    ):
        print("[INFO] Initializing Rule Engine with Meta-Learning Architecture...")
        self.ocr_data = ocr_data
        self.image_metadata = image_metadata
        self.product_metadata = product_metadata
        self.api_client = api_client
        self.plan_cache_path = Path(plan_cache_path)

        with open(rules_path, 'r', encoding='utf-8') as f:
            self.all_rules = json.load(f)

        self.available_tools = {
            "find_text_blocks": self.tool_find_text_blocks,
            "get_font_size": self.tool_get_font_size,
            "get_position": self.tool_get_position,
            "get_product_metadata": self.tool_get_product_metadata,
            "check_proximity": self.tool_check_proximity,
            "check_not_on_panel": self.tool_check_not_on_panel,
            "check_on_panel": self.tool_check_on_panel,
            "get_text_panel": self.tool_get_text_panel,
            "check_text_uppercase": self.tool_check_text_uppercase,
            "check_text_underlined": self.tool_check_text_underlined,
        }

        self.tools_definition_string = self._build_tools_definition()
        self.plan_cache = self._load_cache()

        self.schema_analyzer = SchemaAnalyzer(api_client)
        self.plan_validator = PlanValidator(self.schema_analyzer)

        print(f"[INFO] Rule Engine ready. {len(self.plan_cache) - 1} plans loaded from cache.")

    def _build_tools_definition(self) -> str:
        tool_defs = []
        for name, func in self.available_tools.items():
            doc = func.__doc__.strip().split('\n')
            description = doc[0]
            params_lines = doc[1:]
            param_defs = {}
            for line in params_lines:
                if ':' in line:
                    parts = line.strip().split(':', 1)
                    param_defs[parts[0].strip()] = parts[1].strip()
            tool_defs.append({"name": name, "description": description, "parameters": param_defs})
        return json.dumps(tool_defs, indent=2)

    def _load_cache(self) -> Dict:
        if self.plan_cache_path.exists():
            with open(self.plan_cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                cached_version = cache_data.get("_prompt_version", "unknown")
                if cached_version != PROMPT_VERSION:
                    print(f"[INFO] Prompt version changed ({cached_version} -> {PROMPT_VERSION}). Clearing cache.")
                    return {"_prompt_version": PROMPT_VERSION}
                return cache_data
        return {"_prompt_version": PROMPT_VERSION}

    def _save_cache(self) -> None:
        with open(self.plan_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.plan_cache, f, indent=2, ensure_ascii=False)

    def tool_find_text_blocks(self, keyword: str = "", fuzzy: bool = False) -> List[Dict]:
        """Finds all text blocks containing a keyword. Returns a list of dictionaries, each with 'id' and 'text'.
        keyword: The text to search for (required)
        fuzzy: If True, uses fuzzy matching (optional)"""
        if not keyword:
            return []

        results = []
        keyword_stripped = keyword.strip()
        is_short_text = len(keyword_stripped) <= 3  # TM, 3+, ®, ©, etc.

        for paragraph in self.ocr_data:
            text = paragraph.get('Text', '')

            if is_short_text:
                if keyword_stripped in text:
                    results.append({'id': paragraph.get('Paragraph ID'), 'text': keyword_stripped, 'full_text': text})
                    continue

                if keyword_stripped.lower() in text.lower():
                    text_lower = text.lower()
                    keyword_lower = keyword_stripped.lower()
                    idx = text_lower.find(keyword_lower)
                    if idx != -1:
                        matched_text = text[idx:idx + len(keyword_stripped)]
                        results.append({'id': paragraph.get('Paragraph ID'), 'text': matched_text, 'full_text': text})
                        continue

                escaped_keyword = re.escape(keyword_stripped)
                pattern = r'\b' + escaped_keyword + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    match = re.search(pattern, text, re.IGNORECASE)
                    results.append({'id': paragraph.get('Paragraph ID'), 'text': match.group(0), 'full_text': text})
                    continue
            else:
                normalized_text = self._normalize_text_for_search(text)
                normalized_keyword = self._normalize_text_for_search(keyword)

                is_long_keyword = len(keyword_stripped) > 20
                use_fuzzy = fuzzy or is_long_keyword

                if use_fuzzy:
                    if self._fuzzy_match(normalized_keyword, normalized_text):
                        relevant_text = self._extract_relevant_text(text, keyword)
                        results.append({'id': paragraph.get('Paragraph ID'), 'text': relevant_text, 'full_text': text})
                    elif is_long_keyword:
                        for ratio in [0.7, 0.5, 0.3]:
                            substring_len = int(len(keyword_stripped) * ratio)
                            if substring_len < 10:
                                break
                            substring = keyword_stripped[:substring_len]
                            if substring.lower() in text.lower():
                                relevant_text = self._extract_relevant_text(text, substring)
                                results.append({'id': paragraph.get('Paragraph ID'), 'text': relevant_text, 'full_text': text})
                                break
                else:
                    if normalized_keyword in normalized_text:
                        relevant_text = self._extract_relevant_text(text, keyword)
                        results.append({'id': paragraph.get('Paragraph ID'), 'text': relevant_text, 'full_text': text})

        return results

    def _normalize_text_for_search(self, text: str) -> str:
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace('§', 's').replace('¿', '?')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _fuzzy_match(self, keyword: str, text: str, threshold: float = 0.8) -> bool:
        keyword_words = keyword.lower().split()
        text_lower = text.lower()
        matches = sum(1 for word in keyword_words if word in text_lower)
        match_ratio = matches / len(keyword_words) if keyword_words else 0
        return match_ratio >= threshold

    def _extract_relevant_text(self, full_text: str, keyword: str, context_words: int = 10) -> str:
        words = full_text.split()
        keyword_words = keyword.lower().split()

        start_idx = 0
        for i, word in enumerate(words):
            if any(kw in word.lower() for kw in keyword_words):
                start_idx = max(0, i - context_words)
                break

        end_idx = min(len(words), start_idx + len(keyword_words) + 2 * context_words)
        relevant_words = words[start_idx:end_idx]
        result = ' '.join(relevant_words)

        if len(result) > 200:
            sentences = result.split('. ')
            for sentence in sentences:
                if any(kw.lower() in sentence.lower() for kw in keyword_words):
                    return sentence.strip() + ('.' if not sentence.endswith('.') else '')
            return result[:200] + '...'

        return result

    def tool_get_font_size(self, block_id: int) -> float | None:
        """Gets the font size (in mm) for a specific text block ID.
        block_id: The ID of the text block (required)"""
        if block_id is None:
            return None
        for para in self.ocr_data:
            if para.get("Paragraph ID") == block_id:
                try:
                    font_size_val = para.get('Font Size', '0.0')
                    if isinstance(font_size_val, str):
                        return float(font_size_val.replace(' mm', ''))
                    return float(font_size_val)
                except (ValueError, TypeError):
                    return None
        return None

    def tool_get_position(self, block_id: int) -> Dict | None:
        """Gets the relative position for a specific text block ID. Returns dict with 'ymax_relative' key.
        block_id: The ID of the text block (required)"""
        if block_id is None:
            return None
        for para in self.ocr_data:
            if para.get("Paragraph ID") == block_id:
                bbox = para.get('Bounding Box', {})
                image_height = self.image_metadata.get('image_height')
                if image_height is not None and image_height > 0:
                    ymax_rel = bbox.get('ymax', 0) / image_height
                    ymin_rel = bbox.get('ymin', 0) / image_height
                    return {"ymax_relative": ymax_rel, "ymin_relative": ymin_rel}
                else:
                    print(f"[WARN] Image height not available for position calculation for block {block_id}.")
                    return None
        return None

    def tool_get_product_metadata(self, property_name: str = None, **kwargs) -> Any:
        """Gets a property value from the product metadata, such as 'item_count', 'packaging_type', 'country'.
        property_name: The name of the property to retrieve (required)"""
        key = property_name if property_name is not None else kwargs.get('property')
        if key:
            return self.product_metadata.get(key)
        return None

    def tool_check_proximity(self, block_id_1: int, block_id_2: int, max_distance_mm: float = 50.0) -> Dict:
        """Checks if two text blocks are close to each other. Returns dict with 'is_close' and 'distance_mm'.
        block_id_1: ID of first text block (required)
        block_id_2: ID of second text block (required)
        max_distance_mm: Maximum distance in mm to consider 'close' (optional, default 50mm)"""
        pos1 = self.tool_get_position(block_id_1)
        pos2 = self.tool_get_position(block_id_2)
        if not pos1 or not pos2:
            return {"is_close": False, "distance_mm": None, "reason": "One or both blocks not found"}

        vertical_distance_rel = abs(pos1['ymin_relative'] - pos2['ymin_relative'])
        image_height_mm = self.image_metadata.get('image_height', 1000) * 0.264583
        distance_mm = vertical_distance_rel * image_height_mm
        is_close = distance_mm <= max_distance_mm
        return {
            "is_close": is_close,
            "distance_mm": round(distance_mm, 2),
            "reason": f"Distance is {distance_mm:.2f}mm" + (" (within limit)" if is_close else " (exceeds limit)"),
        }

    def tool_check_not_on_panel(self, keyword: str, forbidden_panel: str) -> bool:
        """Check if text is NOT on a specific panel (e.g., bottom, back).
        keyword: Text to search for (required)
        forbidden_panel: Panel type to avoid - 'bottom', 'back', 'side', 'top' (required)"""
        blocks = self.tool_find_text_blocks(keyword)
        if not blocks:
            return True
        for block in blocks:
            block_id = block.get('id')
            for para in self.ocr_data:
                if para.get('Paragraph ID') == block_id:
                    panel = para.get('panel', 'unknown')
                    if panel == forbidden_panel:
                        return False
        return True

    def tool_check_on_panel(self, keyword: str, required_panel: str) -> bool:
        """Check if text IS on a specific panel (e.g., pdp, front).
        keyword: Text to search for (required)
        required_panel: Panel type required - 'pdp', 'bottom', 'back', 'side', 'top' (required)"""
        blocks = self.tool_find_text_blocks(keyword)
        if not blocks:
            return False
        for block in blocks:
            block_id = block.get('id')
            for para in self.ocr_data:
                if para.get('Paragraph ID') == block_id:
                    panel = para.get('panel', 'unknown')
                    if panel == required_panel:
                        return True
        return False

    def tool_get_text_panel(self, keyword: str) -> str:
        """Get which panel a text is located on.
        keyword: Text to search for (required)
        Returns: Panel name ('pdp', 'bottom', 'back', 'side', 'top', 'unknown', 'not_found')"""
        blocks = self.tool_find_text_blocks(keyword)
        if not blocks:
            return 'not_found'
        block_id = blocks[0].get('id')
        for para in self.ocr_data:
            if para.get('Paragraph ID') == block_id:
                return para.get('panel', 'unknown')
        return 'unknown'

    def tool_check_text_uppercase(self, keyword: str) -> bool:
        """Check if the found text is in UPPERCASE format.
        keyword: Text to search for (required)
        Returns: True if text is found AND all alphabetic characters are uppercase, False otherwise"""
        blocks = self.tool_find_text_blocks(keyword)
        if not blocks:
            return False
        for block in blocks:
            block_id = block.get('id')
            for para in self.ocr_data:
                if para.get('Paragraph ID') == block_id:
                    actual_text = para.get('Text', '')
                    alphabetic_chars = [c for c in actual_text if c.isalpha()]
                    if alphabetic_chars:
                        return all(c.isupper() for c in alphabetic_chars)
        return False

    def tool_check_text_underlined(self, keyword: str) -> bool:
        """Check if the found text has underline formatting.
        keyword: Text to search for (required)
        Returns: False (OCR data does not provide underline formatting information)"""
        blocks = self.tool_find_text_blocks(keyword)
        if not blocks:
            return False
        print(f"[WARNING] Underline validation requested for '{keyword}' but OCR does not provide underline metadata")
        return False

    def _create_cache_key(self, condition_string: str, required_text: List[Dict]) -> str:
        import hashlib

        text_signature = ""
        if required_text:
            text_values = []
            for item in required_text[:3]:
                if isinstance(item, dict) and 'text' in item:
                    text_values.append(item['text'][:50])
            text_signature = hashlib.md5("|".join(text_values).encode()).hexdigest()[:8]

        if text_signature:
            return f"{condition_string}___{text_signature}"
        return condition_string

    def _validate_plan(self, plan: Dict) -> Tuple[bool, str]:
        if not plan:
            return False, "Plan is empty"
        if "plan" not in plan:
            return False, "Plan missing 'plan' key"
        if not isinstance(plan["plan"], list):
            return False, "'plan' must be a list"
        if len(plan["plan"]) == 0:
            return False, "Plan has no steps"

        for i, step in enumerate(plan["plan"]):
            if "tool" not in step:
                return False, f"Step {i} missing 'tool' field"

            tool_name = step["tool"]
            if tool_name not in self.available_tools and tool_name != "assert":
                return False, f"Step {i} uses unknown tool: {tool_name}"

            if tool_name == "assert":
                if "condition" not in step:
                    return False, f"Assert step {i} missing 'condition' field"
            else:
                if "params" not in step:
                    return False, f"Step {i} missing 'params' field"

        has_assert = any(step.get("tool") == "assert" for step in plan["plan"])
        if not has_assert:
            return False, "Plan has no assertion step"

        return True, "Valid"

    def _create_rule_based_plan(self, condition_string: str, required_text: List[Dict]) -> Dict | None:
        condition_lower = condition_string.lower()

        if 'text_available' in condition_lower and '= true' in condition_lower and required_text:
            keywords = [t.get('text', '')[:50] for t in required_text if isinstance(t, dict) and t.get('text')]
            if keywords:
                if len(keywords) == 1:
                    return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keywords[0]}, "store_as": "blocks"}, {"tool": "assert", "condition": "len($blocks) > 0"}]}

                plan_steps = []
                for i, kw in enumerate(keywords):
                    plan_steps.append({"tool": "find_text_blocks", "params": {"keyword": kw}, "store_as": f"blocks{i}"})
                or_conditions = " or ".join([f"len($blocks{i}) > 0" for i in range(len(plan_steps))])
                plan_steps.append({"tool": "assert", "condition": or_conditions})
                return {"plan": plan_steps}

        if 'position' in condition_lower and 'any_panel' in condition_lower and required_text:
            keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
            if keyword:
                return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keyword[:50]}, "store_as": "blocks"}, {"tool": "assert", "condition": "len($blocks) > 0"}]}

        if 'text_format' in condition_lower and required_text:
            keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
            if not keyword and 'made_in' in condition_lower:
                keyword = 'MADE IN'
            if keyword:
                return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keyword[:50]}, "store_as": "blocks"}, {"tool": "assert", "condition": "len($blocks) > 0"}]}

        if 'company_name_and_address_available' in condition_lower and required_text:
            keywords = [t.get('text', '')[:50] for t in required_text if isinstance(t, dict) and t.get('text')]
            if keywords:
                if len(keywords) == 1:
                    return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keywords[0]}, "store_as": "blocks"}, {"tool": "assert", "condition": "len($blocks) > 0"}]}
                plan_steps = []
                for i, kw in enumerate(keywords):
                    plan_steps.append({"tool": "find_text_blocks", "params": {"keyword": kw}, "store_as": f"blocks{i}"})
                or_conditions = " or ".join([f"len($blocks{i}) > 0" for i in range(len(plan_steps))])
                plan_steps.append({"tool": "assert", "condition": or_conditions})
                return {"plan": plan_steps}

        if 'address_location_not_bottom_panel' in condition_lower and required_text:
            keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
            if keyword:
                return {"plan": [{"tool": "check_not_on_panel", "params": {"keyword": keyword[:50], "forbidden_panel": "bottom"}, "store_as": "not_on_bottom"}, {"tool": "assert", "condition": "$not_on_bottom == True"}]}

        if 'close_proximity' in condition_lower or 'proximity' in condition_lower:
            keywords = [t.get('text', '')[:50] for t in required_text if isinstance(t, dict) and t.get('text')]
            if keywords:
                if len(keywords) >= 2:
                    return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keywords[0]}, "store_as": "blocks1"}, {"tool": "find_text_blocks", "params": {"keyword": keywords[1]}, "store_as": "blocks2"}, {"tool": "assert", "condition": "len($blocks1) > 0 and len($blocks2) > 0"}]}
                return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keywords[0]}, "store_as": "blocks"}, {"tool": "assert", "condition": "len($blocks) > 0"}]}

        if 'same_panel_as' in condition_lower and required_text:
            keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
            if keyword:
                return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keyword[:50]}, "store_as": "blocks"}, {"tool": "assert", "condition": "len($blocks) > 0"}]}

        if 'text_underlined' in condition_lower or 'underlined' in condition_lower:
            if required_text:
                keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
                if keyword:
                    return {"plan": [{"tool": "check_text_underlined", "params": {"keyword": keyword[:50]}, "store_as": "is_underlined"}, {"tool": "assert", "condition": "$is_underlined == True"}]}

        if 'text_uppercase' in condition_lower or 'uppercase' in condition_lower or 'all_caps' in condition_lower:
            if required_text:
                keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
                if keyword:
                    return {"plan": [{"tool": "check_text_uppercase", "params": {"keyword": keyword[:50]}, "store_as": "is_uppercase"}, {"tool": "assert", "condition": "$is_uppercase == True"}]}

        if any(keyword in condition_lower for keyword in ['height >=', 'size >=', 'height >', 'size >']):
            match = re.search(r'(>=|>)\s*(\d+\.?\d*)\s*mm', condition_string)
            if match and required_text:
                operator, size_value = match.groups()
                keywords = [t.get('text', '')[:50] for t in required_text if isinstance(t, dict) and t.get('text')]
                if keywords:
                    keyword = keywords[0]
                    return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keyword}, "store_as": "blocks"}, {"tool": "assert", "condition": f"len($blocks) > 0 and get_font_size(block_id=$blocks[0]['id']) {operator} {size_value}"}]}

        if '=' in condition_string and required_text and 'text_available' not in condition_lower:
            keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
            if keyword and len(keyword) > 5:
                return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keyword}, "store_as": "blocks"}, {"tool": "assert", "condition": "len($blocks) > 0"}]}

        if 'language' in condition_lower and required_text and '[' not in condition_string:
            keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
            if keyword:
                return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keyword[:50]}, "store_as": "blocks"}, {"tool": "assert", "condition": "len($blocks) > 0"}]}

        if 'language' in condition_lower and '[' in condition_string and required_text:
            keywords = [t.get('text', '')[:50] for t in required_text if isinstance(t, dict) and t.get('text')]
            if len(keywords) >= 2:
                return {"plan": [{"tool": "find_text_blocks", "params": {"keyword": keywords[0]}, "store_as": "blocks1"}, {"tool": "find_text_blocks", "params": {"keyword": keywords[1]}, "store_as": "blocks2"}, {"tool": "assert", "condition": "len($blocks1) > 0 or len($blocks2) > 0"}]}

        if 'except' in condition_lower and 'bottom' in condition_lower and required_text:
            keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
            if keyword:
                return {"plan": [{"tool": "check_not_on_panel", "params": {"keyword": keyword[:50], "forbidden_panel": "bottom"}, "store_as": "not_on_bottom"}, {"tool": "assert", "condition": "$not_on_bottom == True"}]}

        if any(keyword in condition_lower for keyword in ['= pdp', '= front', 'principal display']) and required_text:
            keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
            if keyword:
                return {"plan": [{"tool": "check_on_panel", "params": {"keyword": keyword[:50], "required_panel": "pdp"}, "store_as": "on_pdp"}, {"tool": "assert", "condition": "$on_pdp == True"}]}

        if 'back' in condition_lower and 'panel' in condition_lower and required_text:
            keyword = required_text[0].get('text', '') if isinstance(required_text[0], dict) else ''
            if keyword:
                return {"plan": [{"tool": "check_on_panel", "params": {"keyword": keyword[:50], "required_panel": "back"}, "store_as": "on_back"}, {"tool": "assert", "condition": "$on_back == True"}]}

        return None

    def _generate_plan_with_llm(self, condition_string: str, required_text: List[Dict], model_name: str, max_retries: int = 3) -> Dict | None:
        print(f"\n[INFO] Unrecognized rule: '{condition_string}'. Asking LLM to create a plan...")

        if not required_text or len(required_text) == 0:
            print(f"[ERROR] Cannot create plan: required_text is empty for condition '{condition_string}'")
            print(f"[ERROR] This rule needs text to validate but none was provided in legal_rules.json")
            print(f"[ERROR] Please check legal_term.json and ensure text extraction is working correctly")
            return None

        rule_based_plan = self._create_rule_based_plan(condition_string, required_text)
        if rule_based_plan:
            print("[SUCCESS] Created plan using rule-based patterns!")
            cache_key = self._create_cache_key(condition_string, required_text)
            self.plan_cache[cache_key] = rule_based_plan
            self._save_cache()
            return rule_based_plan

        analysis = self.schema_analyzer.analyze_requirement(condition_string, required_text)
        print(f"[ANALYSIS] {analysis['reasoning']}")
        print(f"[ANALYSIS] Required tools: {', '.join(analysis['tools_needed'])}")

        ocr_summary = json.dumps([{"id": p.get('Paragraph ID'), "text": p.get('Text', '')[:50] + "..."} for p in self.ocr_data[:10]])
        keywords_to_search = []
        for text_entry in required_text:
            if isinstance(text_entry, dict) and 'text' in text_entry:
                keywords_to_search.append(text_entry['text'])

        prompt = PROMPT_TEMPLATE.format(
            tools_definition_string=self.tools_definition_string,
            ocr_summary=ocr_summary,
            product_metadata_json=json.dumps(self.product_metadata),
            condition_string=condition_string,
            required_text_json=json.dumps(required_text, ensure_ascii=False) if required_text else '[]',
            keywords_hint=", ".join([f'\"{k[:30]}...\"' for k in keywords_to_search[:3]]) if keywords_to_search else 'No specific text required',
        )

        if analysis['validation_types']:
            prompt += f"\n\n**IMPORTANT:** This requirement involves {', '.join(analysis['validation_types'])} validation. Make sure your plan includes all necessary tools: {', '.join(analysis['tools_needed'])}."

        use_structured_output = (
            'gpt-4o' in model_name.lower()
            or 'gpt-3.5' in model_name.lower()
            or ('llama' in model_name.lower() and 'fireworks.ai' in str(self.api_client.base_url))
        )

        for attempt in range(max_retries):
            response_content = call_api_with_retry(
                self.api_client,
                prompt,
                model_name=model_name,
                use_json_schema=use_structured_output,
            )
            if not response_content:
                continue

            try:
                json_block = extract_json_block(response_content)
                plan = json.loads(json_block) if json_block else json.loads(response_content)

                is_valid, validation_msg = self._validate_plan(plan)
                if not is_valid:
                    print(f"[WARN] Syntax validation failed (attempt {attempt + 1}/{max_retries}): {validation_msg}")
                    if attempt < max_retries - 1:
                        print("[INFO] Retrying plan generation...")
                        continue
                    return None

                is_complete, completeness_msg = self.plan_validator.validate_plan_completeness(condition_string, required_text, plan)
                if not is_complete:
                    print(f"[WARN] Plan completeness check failed (attempt {attempt + 1}/{max_retries}): {completeness_msg}")
                    if attempt < max_retries - 1:
                        prompt += f"\n\n**FEEDBACK FROM PREVIOUS ATTEMPT:** {completeness_msg}. Please generate a complete plan."
                        print("[INFO] Retrying with feedback...")
                        continue

                    print(f"[WARN] Failed to generate complete plan after {max_retries} attempts. Trying simplified approach...")
                    simplified_prompt = f"""Generate a simple execution plan to check: {condition_string}

                    Available tools: {json.dumps(list(self.available_tools.keys()))}

                    Return ONLY valid JSON with format:
                    {{
                      \"plan\": [
                        {{\"tool\": \"find_text_blocks\", \"params\": {{\"keyword\": \"text to find\"}}, \"store_as\": \"blocks\"}},
                        {{\"tool\": \"assert\", \"condition\": \"len($blocks) > 0\"}}
                      ]
                    }}"""

                    response = call_api_with_retry(self.api_client, simplified_prompt, model_name, max_retries=1, use_json_schema=use_structured_output)
                    if response:
                        try:
                            json_block_2 = extract_json_block(response)
                            plan = json.loads(json_block_2) if json_block_2 else json.loads(response)
                            if plan and 'plan' in plan:
                                print("[SUCCESS] Simplified plan generated!")
                                cache_key = self._create_cache_key(condition_string, required_text)
                                self.plan_cache[cache_key] = plan
                                self._save_cache()
                                return plan
                        except Exception:
                            pass
                    print("[ERROR] All attempts failed. Cannot generate plan.")
                    return None

                cache_key = self._create_cache_key(condition_string, required_text)
                self.plan_cache[cache_key] = plan
                self._save_cache()
                print("[SUCCESS] Plan generated and validated successfully!")
                return plan

            except (json.JSONDecodeError, TypeError) as e:
                print(f"[ERROR] Invalid JSON (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                return None

        return None

    def _execute_plan(self, plan: Dict, condition_str: str) -> Tuple[bool, str, Dict]:
        safe_globals = {
            "__builtins__": {
                "len": len,
                "str": str,
                "float": float,
                "int": int,
                "any": any,
                "all": all,
                "abs": abs,
                "round": round,
                "True": True,
                "False": False,
                "None": None,
            },
            **self.available_tools,
        }
        step_results = {}
        execution_trace = []

        for step_idx, step in enumerate(plan.get('plan', [])):
            tool_name = step.get('tool')
            params = step.get('params', {}).copy()

            for key, value in params.items():
                if isinstance(value, str) and value.startswith('$'):
                    var_name = value[1:]
                    try:
                        resolved_value = eval(var_name, {"__builtins__": {}}, step_results)
                        params[key] = resolved_value
                    except (IndexError, KeyError) as e:
                        print(f"[WARN] Variable resolution failed for '{value}': {e}. Trying fallback strategies...")

                        if '[0]' in var_name:
                            base_var = var_name.split('[')[0]
                            if base_var in step_results:
                                array_value = step_results[base_var]
                                if isinstance(array_value, list) and len(array_value) == 0:
                                    print(f"[INFO] Array '{base_var}' exists but is empty. Text not found.")
                                    context = {
                                        'status': 'FAIL',
                                        'condition': condition_str,
                                        'assertion_evaluated': f"len({base_var}) > 0",
                                        'variables_used': {base_var: '[]'},
                                        'expected_text': [],
                                        'text_blocks_found': 0,
                                    }
                                    return False, 'Required text not found on packaging', context

                        if "['id']" in var_name:
                            print(f"[WARN] Skipping step {step_idx} due to empty array access.")
                            continue

                        context = {
                            'error_type': 'Variable Resolution Error',
                            'details': str(e),
                            'available_variables': list(step_results.keys()),
                            'execution_trace': execution_trace,
                        }
                        return False, f"Technical error: Failed to process variable '{value}': {e}", context
                    except Exception as e:
                        context = {
                            "error": str(e),
                            "step": step_idx,
                            "variable": var_name,
                            "available_vars": list(step_results.keys()),
                            "trace": execution_trace,
                        }
                        return False, f"Failed to process variable '{value}': {e}", context

            if tool_name == 'assert':
                condition_code = step.get('condition', 'False').replace('$', '')
                eval_locals = step_results
                try:
                    result = eval(condition_code, safe_globals, eval_locals)
                    context = {
                        'condition': condition_str,
                        'assertion': condition_code,
                        'result': result,
                        'variables': {k: str(v)[:200] for k, v in step_results.items()},
                        'trace': execution_trace,
                    }

                    if result:
                        return True, f"Condition met: {condition_code}", context
                    return False, f"Condition not met: {condition_code}", context
                except Exception as e:
                    context = {"error": str(e), "condition": condition_code, "variables": step_results, "trace": execution_trace}
                    return False, f"Error during evaluation: {e}", context

            if tool_name in self.available_tools:
                try:
                    result = self.available_tools[tool_name](**params)
                    if 'store_as' in step:
                        step_results[step['store_as']] = result
                    execution_trace.append({"step": step_idx, "tool": tool_name, "params": params, "result": str(result)[:200]})
                except Exception as e:
                    context = {"error": str(e), "step": step_idx, "tool": tool_name, "params": params, "trace": execution_trace}
                    return False, f"Error during tool execution '{tool_name}': {e}", context
            else:
                context = {"error": f"Unknown tool: {tool_name}", "trace": execution_trace}
                return False, f"Unknown tool: {tool_name}", context

        context = {"error": "No assertion found", "trace": execution_trace}
        return False, "Plan finished without a final assertion", context

    def _generate_natural_explanation(self, condition_str: str, is_compliant: bool, context: Dict, required_text: list = None) -> str:
        try:
            status = 'PASSED' if is_compliant else 'FAILED'
            variables_summary = ''
            if 'variables' in context:
                vars_dict = context['variables']
                for key, value in vars_dict.items():
                    value_str = str(value)[:150]
                    variables_summary += f"- {key}: {value_str}\n"

            expected_summary = ''
            if required_text:
                expected_summary = 'Expected text on packaging:\n'
                for item in required_text[:3]:
                    if isinstance(item, dict):
                        lang = item.get('lang', 'Unknown')
                        text = item.get('text', '')[:100]
                        expected_summary += f"- [{lang}] {text}\n"

            prompt = f"""You are a packaging compliance expert explaining validation results to non-technical stakeholders.

            **Validation Rule:** {condition_str}
            **Result:** {status}

            **Technical Details:**
            {variables_summary}

            {expected_summary}

            **Task:** Write a clear, concise explanation (2-3 sentences) in natural language that:
            1. Explains what was checked
            2. States whether it passed or failed
            3. If failed, explains why in simple terms
            4. Avoids technical jargon (no "blocks", "variables", "assertions")

            Write the explanation now (2-3 sentences only):"""

            response = call_api_with_retry(
                self.api_client,
                prompt,
                "accounts/fireworks/models/llama-v3p1-70b-instruct",
                max_retries=1,
                use_json_schema=False,
            )
            if response:
                explanation = response.strip().replace('**', '').replace('*', '')
                if len(explanation) > 500:
                    sentences = explanation.split('. ')
                    explanation = '. '.join(sentences[:3]) + '.'
                return explanation
        except Exception as e:
            print(f"[WARN] Failed to generate natural explanation: {e}")

        if is_compliant:
            return f"The requirement '{condition_str}' was successfully validated and met."
        return f"The requirement '{condition_str}' was not met. Please review the technical details below."

    def _format_human_reason(self, condition_str: str, is_compliant: bool, raw_reason: str, context: Dict, required_text: list = None) -> Dict:
        natural_explanation = self._generate_natural_explanation(condition_str, is_compliant, context, required_text)

        if is_compliant:
            return {
                'reason': 'Requirement met',
                'evidence': {
                    'status': 'PASS',
                    'details': raw_reason,
                    'human_explanation': natural_explanation,
                    'context': context,
                    'expected_text': required_text if required_text else [],
                },
            }

        if 'Failed to process variable' in raw_reason:
            return {
                'reason': f"Technical error: {raw_reason}",
                'evidence': {
                    'error_type': 'Variable Resolution Error',
                    'details': context.get('error', ''),
                    'human_explanation': natural_explanation,
                    'available_variables': context.get('available_vars', []),
                    'execution_trace': context.get('trace', []),
                    'expected_text': required_text if required_text else [],
                },
            }

        if 'Error during evaluation' in raw_reason or 'Error during tool execution' in raw_reason:
            return {
                'reason': f"Technical error: {raw_reason}",
                'evidence': {
                    'error_type': 'Execution Error',
                    'details': context.get('error', ''),
                    'human_explanation': natural_explanation,
                    'step': context.get('step', 'unknown'),
                    'execution_trace': context.get('trace', []),
                    'expected_text': required_text if required_text else [],
                },
            }

        if 'Condition not met' in raw_reason:
            evidence = {
                'status': 'FAIL',
                'condition': condition_str,
                'assertion_evaluated': context.get('assertion', ''),
                'human_explanation': natural_explanation,
                'variables_used': context.get('variables', {}),
                'expected_text': required_text if required_text else [],
            }
            return {'reason': f"Condition not met: {condition_str}", 'evidence': evidence}

        return {
            'reason': f"Validation failed: {raw_reason}",
            'evidence': {'raw_reason': raw_reason, 'context': context, 'expected_text': required_text if required_text else []},
        }

    def _find_child_rules(self, parent_id: str) -> List[Dict]:
        child_rules = []
        for rule_container in self.all_rules:
            for rule_item in rule_container.get('item', []):
                if rule_item.get('parent_rule') == parent_id:
                    child_rules.append(rule_item)
        return child_rules

    def _get_text_from_siblings(self, rule_id: str) -> List[Dict]:
        parent_id = None
        for rule_container in self.all_rules:
            for rule_item in rule_container.get('item', []):
                if rule_item.get('item_no') == rule_id:
                    parent_id = rule_item.get('parent_rule')
            if parent_id:
                break

        if not parent_id:
            return []

        all_text = []
        for rule_container in self.all_rules:
            for rule_item in rule_container.get('item', []):
                if rule_item.get('item_no') == parent_id and 'rules' in rule_item:
                    for rule_data in rule_item['rules']:
                        text_entries = rule_data.get('text', [])
                        if text_entries:
                            all_text.extend(text_entries)
                            print(f"[INFO] Child rule {rule_id} inherited {len(text_entries)} text entries from parent {parent_id}")
                            return all_text

        for rule_container in self.all_rules:
            for rule_item in rule_container.get('item', []):
                if rule_item.get('parent_rule') == parent_id and 'rules' in rule_item:
                    for rule_data in rule_item['rules']:
                        text_entries = rule_data.get('text', [])
                        if text_entries:
                            all_text.extend(text_entries)

        if all_text:
            print(f"[INFO] Child rule {rule_id} inherited {len(all_text)} text entries from siblings")

        return all_text

    def validate(self, model_name: str, delay_between_calls: float) -> List[Dict]:
        final_report = []
        last_call_time = 0
        reference_only_rules = []

        total_items = sum(len(rule_container.get('item', [])) for rule_container in self.all_rules)
        with tqdm(total=total_items, desc='Validating all rules') as pbar:
            for rule_container in self.all_rules:
                for rule_item in rule_container.get('item', []):
                    rule_id = rule_item.get('item_no', 'N/A')
                    rule_name = rule_item.get('item_name', 'No Name')

                    if rule_item.get('is_reference_only', False):
                        reference_only_rules.append((rule_id, rule_name, rule_item))
                        pbar.update(1)
                        continue

                    if not rule_item or 'rules' not in rule_item or not rule_item['rules']:
                        pbar.update(1)
                        continue

                    rule_data = rule_item['rules'][0]
                    required_text = rule_data.get('text', [])
                    conditions = rule_data.get('requirement', [])

                    if rule_item.get('is_parent_rule', False) and not required_text:
                        child_rules = self._find_child_rules(rule_id)
                        if child_rules:
                            print(f"[INFO] Skipping parent rule {rule_id} - will aggregate from children later")
                            pbar.update(1)
                            continue

                    if rule_item.get('parent_rule') and not required_text:
                        sibling_text = self._get_text_from_siblings(rule_id)
                        if sibling_text:
                            required_text = sibling_text
                            print(f"[INFO] Rule {rule_id} inherited {len(sibling_text)} text entries from siblings")
                        else:
                            rule_name_lower = str(rule_name).lower()
                            if 'country of origin' in rule_name_lower or 'country' in rule_name_lower:
                                required_text = [{"lang": "English", "text": "MADE IN"}]
                                print(f"[INFO] Rule {rule_id} using generic 'MADE IN' text for Country of Origin")

                    rule_status = {"rule_id": rule_id, "rule_name": rule_name, "is_compliant": True, "details": []}

                    for cond_str in conditions:
                        cache_key = self._create_cache_key(cond_str, required_text)
                        plan = self.plan_cache.get(cache_key)
                        is_from_cache = plan is not None

                        if not plan:
                            time_since_last_call = time.time() - last_call_time
                            if time_since_last_call < delay_between_calls:
                                time.sleep(delay_between_calls - time_since_last_call)
                            plan = self._generate_plan_with_llm(cond_str, required_text, model_name)
                            last_call_time = time.time()

                        if plan:
                            if is_from_cache:
                                print(f"\n[INFO] Using plan from cache for: '{cond_str}' with text hash.")

                            is_compliant, raw_reason, context = self._execute_plan(plan, cond_str)
                            human_readable_result = self._format_human_reason(cond_str, is_compliant, raw_reason, context, required_text)
                            rule_status['details'].append({
                                'condition': cond_str,
                                'status': 'PASS' if is_compliant else 'FAIL',
                                'reason': human_readable_result['reason'],
                                'evidence': human_readable_result.get('evidence'),
                            })
                            if not is_compliant:
                                rule_status['is_compliant'] = False
                        else:
                            rule_status['details'].append({
                                'condition': cond_str,
                                'status': 'FAIL',
                                'reason': 'Cannot create execution plan',
                                'evidence': {'error': 'LLM failed to generate valid plan', 'expected_text': required_text},
                            })
                            rule_status['is_compliant'] = False

                    final_report.append(rule_status)
                    pbar.update(1)

        print("\n[INFO] Processing is_reference_only rules...")
        for rule_id, rule_name, rule_item in reference_only_rules:
            if not rule_item or 'rules' not in rule_item or not rule_item['rules']:
                continue

            rule_data = rule_item['rules'][0]
            conditions = rule_data.get('requirement', [])
            refer_to_pattern = r"refer_to_rules\s*=\s*\[([^\]]+)\]"
            referenced_rules = []
            for cond in conditions:
                match = re.search(refer_to_pattern, cond)
                if match:
                    refs = match.group(1).replace("'", "").replace('"', '').split(',')
                    referenced_rules.extend([r.strip() for r in refs])

            if referenced_rules:
                all_refs_passed = True
                missing_refs = []
                for ref_id in referenced_rules:
                    ref_result = next((r for r in final_report if r.get('rule_id') == ref_id), None)
                    if not ref_result:
                        missing_refs.append(ref_id)
                        all_refs_passed = False
                    elif not ref_result.get('is_compliant'):
                        all_refs_passed = False

                rule_status = {
                    'rule_id': rule_id,
                    'rule_name': rule_name,
                    'is_compliant': all_refs_passed,
                    'details': [{
                        'condition': f"refer_to_rules = {referenced_rules}",
                        'status': 'PASS' if all_refs_passed else 'FAIL',
                        'reason': f"Referenced rule(s) {referenced_rules} {'all passed' if all_refs_passed else 'failed or missing'}",
                        'evidence': {
                            'referenced_rules': referenced_rules,
                            'all_passed': all_refs_passed,
                            'missing_refs': missing_refs if missing_refs else None,
                        },
                    }],
                }
                final_report.append(rule_status)
                print(f"[INFO] Processed is_reference_only rule {rule_id}: {'PASS' if all_refs_passed else 'FAIL'}")

        self._save_cache()
        return final_report


def call_api_with_retry(
    client: OpenAI,
    prompt: str,
    model_name: str,
    max_retries: int = 3,
    use_json_schema: bool = False,

) -> str | None:
    for attempt in range(max_retries):
        try:
            request_params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.0,
            }

            if use_json_schema:
                is_fireworks_llama = client.base_url and "fireworks.ai" in client.base_url and "llama" in model_name.lower()
                is_openai_gpt = "openai.com" in str(client.base_url) and ("gpt-4" in model_name.lower() or "gpt-3.5" in model_name.lower())
                if is_fireworks_llama or is_openai_gpt:
                    request_params["response_format"] = {"type": "json_object"}
                else:
                    print(
                        f"[WARN] JSON schema output not explicitly supported for model {model_name} with this client configuration. Proceeding without response_format."
                    )

            completion = client.chat.completions.create(**request_params)
            return completion.choices[0].message.content

        except RateLimitError:
            wait_time = (2 ** attempt) + 1
            print(f"[WARN] Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except APIError as e:
            print(f"[ERROR] API Error (attempt {attempt + 1}/{max_retries}): {e}")
            if getattr(e, "status_code", None) == 429:
                wait_time = (2 ** attempt) + 1
                time.sleep(wait_time)
            elif getattr(e, "status_code", None) == 400 and 'invalid model ID' in str(e):
                print(f"[FATAL] Invalid model ID '{model_name}'. Please check that this model is available for your API key.")
                return None
            else:
                return None
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
            return None

    print(f"[ERROR] API call failed after {max_retries} attempts.")
    return None


def extract_json_block(text: str) -> str | None:
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace == -1 or last_brace == -1:
            return None
        json_str = text[first_brace:last_brace + 1]

    json_str = json_str.strip()
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    return json_str
