import asyncio
import json
import math
from pathlib import Path
from time import perf_counter

from .runbooks import retrieve_runbooks


async def measure(cases, hybrid):
    top_one = 0
    recalled = 0
    abstained = 0
    latencies = []
    missed = []
    for code, question, expected in cases:
        started = perf_counter()
        rows = await retrieve_runbooks(code, question, hybrid=hybrid)
        latencies.append((perf_counter() - started) * 1000)
        sections = [row["url"].split("#")[-1] for row in rows]
        if expected is None:
            abstained += not sections
            if sections:
                missed.append({"question": question, "expected": None, "observed": sections[0]})
        else:
            hit = bool(sections) and sections[0] == expected
            top_one += hit
            recalled += expected in sections
            if not hit:
                missed.append({"question": question, "expected": expected, "observed": (sections or [None])[0]})
    positives = sum(expected is not None for _, _, expected in cases)
    negatives = len(cases) - positives
    return {
        "top1_correct": f"{top_one}/{positives}",
        "recall_at_3": f"{recalled}/{positives}",
        "abstained": f"{abstained}/{negatives}",
        "p95_ms": round(sorted(latencies)[math.ceil(.95 * len(latencies)) - 1], 1),
        "missed_cases": missed,
    }


async def main():
    cases = json.loads(Path(__file__).with_name("eval_cases.json").read_text())
    print(json.dumps({
        "cases": len(cases),
        "vector": await measure(cases, hybrid=False),
        "hybrid": await measure(cases, hybrid=True),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
