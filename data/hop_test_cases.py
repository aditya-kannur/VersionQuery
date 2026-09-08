"""
Static test fixtures: version-hop test cases with expected hop sequences.
Used for manually verifying src/migration_decomposition.py.
"""

HOP_TEST_CASES = [
    {
        "from_version": "2021-05-13",
        "to_version": "2021-08-16",
        "expected_hops": [("2021-05-13", "2021-08-16")],
    },
    {
        "from_version": "2021-08-16",
        "to_version": "2022-06-28",
        "expected_hops": [("2021-08-16", "2022-02-22"), ("2022-02-22", "2022-06-28")],
    },
    {
        "from_version": "2022-06-28",
        "to_version": "2026-03-11",
        "expected_hops": [
            ("2022-06-28", "2025-09-03"),
            ("2025-09-03", "2026-03-11"),
        ],
    },
]
