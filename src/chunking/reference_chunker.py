# src/chunking/reference_chunker.py
"""
Splits openapi.json into one chunk per endpoint (path + method pair).
Structure confirmed from the actual file: top-level "paths" object,
each key is a path, each path has one or more HTTP method keys.
"""
import json

from src.constants import REFERENCE_AVAILABLE_VERSIONS


def chunk_reference_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        spec = json.load(f)

    chunks = []
    paths = spec.get("paths", {})
    components = spec.get("components", {}).get("schemas", {})

    for path, methods in paths.items():
        for method, operation in methods.items():
            # skip non-HTTP-method keys if any ever show up (defensive)
            if method not in ("get", "post", "patch", "delete", "put"):
                continue

            summary = operation.get("summary", "")
            description = operation.get("description", "")
            tags = operation.get("tags", [])

            # Build a readable text block from the operation's own fields —
            # this is what gets embedded, so it needs to read like a sentence,
            # not raw JSON.
            text_parts = [f"{method.upper()} {path} - {summary}."]
            if description and description != summary:
                text_parts.append(description)

            if "parameters" in operation:
                param_bits = []
                for p in operation["parameters"]:
                    name = p.get("name", "")
                    if not name:
                        continue
                    where = p.get("in", "")
                    required = " (required)" if p.get("required") else ""
                    param_bits.append(f"{name} [{where}]{required}")
                if param_bits:
                    text_parts.append(f"Parameters: {', '.join(param_bits)}.")

            if "requestBody" in operation:
                body_schema = (
                    operation["requestBody"]
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema", {})
                )
                # The body schema is either a $ref to a named component, or
                # (more often in this file) an inline object with its own
                # "properties" — handle both rather than assuming a $ref.
                ref = body_schema.get("$ref", "")
                schema_name = ref.rsplit("/", 1)[-1] if ref else None
                if schema_name and schema_name in components:
                    props = components[schema_name].get("properties", {})
                elif body_schema.get("type") == "object":
                    props = body_schema.get("properties", {})
                else:
                    props = {}

                prop_names = list(props.keys())
                if prop_names:
                    text_parts.append(f"Request body fields: {', '.join(prop_names)}.")
                else:
                    text_parts.append("Accepts a request body.")

            responses = operation.get("responses", {})
            ok_response = responses.get("200") or responses.get("201")
            if ok_response:
                resp_desc = ok_response.get("description", "")
                if resp_desc:
                    text_parts.append(f"Response: {resp_desc}.")

            chunks.append({
                "doc_type": "reference",
                "endpoint": path,
                "method": method,
                "tags": tags,
                "summary": summary,
                # The OpenAPI file documents the current reference surface,
                # which per the PRD's Section 3 limitation only covers these
                # two versions — no reference exists for earlier ones.
                "versions": list(REFERENCE_AVAILABLE_VERSIONS),
                "text": " ".join(text_parts),
            })

    return chunks


if __name__ == "__main__":
    chunks = chunk_reference_file("data/reference/openapi.json")
    print(f"Produced {len(chunks)} chunks from openapi.json")
    with open("data/chunks_reference.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
