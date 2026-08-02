"""
mcp_secondary/tools_benchmarks.py

get_population_benchmarks_tool. Looks up simulated population-level
benchmarks by service_line + ICD-10 code. BENCHMARKS below is a
hand-seeded static table -- there is no real dataset behind this, and
every entry's benchmark_source says so explicitly. Extend BENCHMARKS
with more service_line/icd10_code combos as needed for your test data
(e.g. add entries matching mock_ehr/data.py's patients).
"""

from typing import Dict, Optional

from mcp.server.fastmcp import Context

# service_line -> icd10_code -> {readmission_rate_30d, avg_los_days, benchmark_source}
# NOTE: fully simulated numbers for demo/testing purposes only.
BENCHMARKS: Dict[str, Dict[str, dict]] = {
    "Cardiology": {
        "I21.9": {  # Acute Myocardial Infarction
            "readmission_rate_30d": 0.14,
            "avg_los_days": 4.2,
            "benchmark_source": "simulated",
        },
        "E78.5": {  # Hyperlipidemia
            "readmission_rate_30d": 0.05,
            "avg_los_days": 1.8,
            "benchmark_source": "simulated",
        },
        "I50.9": {  # Congestive Heart Failure
            "readmission_rate_30d": 0.22,
            "avg_los_days": 5.6,
            "benchmark_source": "simulated",
        },
    },
    "Endocrinology": {
        "E11": {  # Diabetes Mellitus
            "readmission_rate_30d": 0.11,
            "avg_los_days": 3.1,
            "benchmark_source": "simulated",
        },
    },
    "Nephrology": {
        "N18.9": {  # Chronic Kidney Disease
            "readmission_rate_30d": 0.18,
            "avg_los_days": 4.9,
            "benchmark_source": "simulated",
        },
    },
    "Pulmonology": {
        "J18.9": {  # Pneumonia
            "readmission_rate_30d": 0.16,
            "avg_los_days": 3.7,
            "benchmark_source": "simulated",
        },
    },
    "General Medicine": {
        "I10": {  # Hypertension
            "readmission_rate_30d": 0.07,
            "avg_los_days": 2.0,
            "benchmark_source": "simulated",
        },
        "I25.10": {  # Coronary Artery Disease
            "readmission_rate_30d": 0.13,
            "avg_los_days": 3.9,
            "benchmark_source": "simulated",
        },
    },
}

_NO_DATA = {
    "readmission_rate_30d": None,
    "avg_los_days": None,
    "benchmark_source": "no data",
}


async def get_population_benchmarks_tool(ctx: Context, service_line: str, icd10_code: str) -> Dict:
    """
    Args:
        ctx: MCP Context (unused -- pure lookup, no I/O).
        service_line: e.g. "Cardiology"
        icd10_code: e.g. "I21.9"

    Returns:
        {readmission_rate_30d, avg_los_days, benchmark_source}
    """
    return BENCHMARKS.get(service_line, {}).get(icd10_code, _NO_DATA)