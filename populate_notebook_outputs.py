#!/usr/bin/env python3
"""
populate_notebook_outputs.py

Executes the code cells of Soft_Computing_Project_Demo_RF_Final.ipynb and populates
their 'outputs' in the notebook JSON so it is 100% pre-rendered and ready for live demonstration.
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout, redirect_stderr


def main():
    nb_path = "Soft_Computing_Project_Demo_RF_Final.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Execution global environment
    exec_env = {
        "__name__": "__main__",
        "__file__": nb_path
    }

    exec_count = 1
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue

        source_code = "".join(cell["source"])
        print(f"Executing cell {exec_count}...")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        outputs = []
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(source_code, exec_env)

            out_text = stdout_buf.getvalue()
            err_text = stderr_buf.getvalue()

            if out_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in out_text.splitlines()]
                })
            if err_text:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": [line + "\n" for line in err_text.splitlines()]
                })

            cell["execution_count"] = exec_count
            cell["outputs"] = outputs
            exec_count += 1
        except Exception as e:
            print(f"Error in cell {exec_count}: {e}")
            outputs.append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": [str(e)]
            })
            cell["outputs"] = outputs
            break

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully executed and populated outputs in {nb_path}.")


if __name__ == "__main__":
    main()
