"""Step 3 - exploratory data analysis (figures -> reports/figures, tables -> reports/results)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coxkan_churn import data, eda  # noqa: E402

print(json.dumps(eda.run(data.load_model_table()), indent=2, default=str)[:3000])
