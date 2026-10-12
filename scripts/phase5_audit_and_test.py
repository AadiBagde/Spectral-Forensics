"""Stage 1 and 4: Project Audit and Test Execution (Hardened)."""

import os
import sys
from pathlib import Path
import subprocess

def check_file(path_obj):
    return path_obj.exists()

def main():
    # Resolve the project root portably from the script location
    root = Path(__file__).resolve().parent.parent
    
    expected_files = [
        "docs/final_project_report.md",
        "reports/final_project_report.pdf",
        "docs/final_demo_guide.md",
        "docs/final_presentation_outline.md",
        "datasets/processed/feature_schema.json",
        "datasets/processed/verified_train_features.csv",
        "datasets/processed/verified_test_features.csv",
        "models/svm_v1/model.joblib",
        "experiments/phase2/final_evaluation_report.json",
        "experiments/phase4/baseline/baseline_record.json",
        "experiments/phase4/model_comparison/final_comparison.json",
        "app.py"
    ]
    
    audit_results = []
    missing_critical = False
    for f in expected_files:
        exists = check_file(root / f)
        audit_results.append((f, exists))
        if not exists:
            missing_critical = True
            
    # Write Audit MD
    md = "# Final Project Audit\n\n"
    md += "## File Verification\n"
    for f, exists in audit_results:
        status = "✅ Found" if exists else "❌ Missing"
        md += f"- `{f}`: {status}\n"
        
    md += "\n## Manual Check Limitations\n"
    md += "The automated audit verified the existence of the expected code, documentation, and model artifacts.\n"
    md += "However, **it cannot natively simulate or verify browser UI state, CSV downloads, or visual correctness of PDF formatting.** "
    md += "These UI components require human validation as instructed in the demonstration guide.\n"
        
    print("Running tests with Python executable:", sys.executable)
    # Run tests with the active python interpreter
    test_cmd = [sys.executable, "-m", "pytest", "-v"]
    test_result = subprocess.run(
        test_cmd,
        cwd=str(root),
        capture_output=True,
        text=True
    )
    
    md += "\n## Test Execution (Stage 4 Final Regression)\n"
    md += f"**Command Executed:** `{' '.join(test_cmd)}`\n"
    md += f"**Exit Code:** {test_result.returncode}\n\n"
    md += "```text\n"
    md += test_result.stdout
    if test_result.stderr:
        md += "\n" + test_result.stderr
    md += "```\n"
    
    out_path = root / "docs" / "final_project_audit.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
        
    print(f"Audit written to {out_path}")
    print(f"Test Exit Code: {test_result.returncode}")
    
    if missing_critical:
        print("ERROR: Missing critical artifacts. Exit 1.")
        sys.exit(1)
        
    if test_result.returncode != 0:
        print("ERROR: Pytest failed. Exit 1.")
        sys.exit(1)
        
    print("Audit and Tests passed.")

if __name__ == "__main__":
    main()
