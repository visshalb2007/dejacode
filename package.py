"""
Packaging script for Déjà Code project deliverable.
Creates a portable, self-contained zip archive excluding cache, venv, and git internals.
"""
import os
import zipfile

def create_package(output_zip="dejacode.zip"):
    exclude_dirs = {".git", "__pycache__", ".pytest_cache", "venv", ".idea", ".vscode", ".streamlit"}
    include_data = {"questions.csv", "eval_set.csv"}

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for f in files:
                p = os.path.normpath(os.path.join(root, f))
                if p == output_zip or p.endswith(".pyc") or ".git" in p:
                    continue
                if p.startswith("data" + os.sep) and f not in include_data:
                    continue
                zf.write(p, p)

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"Created {output_zip} ({size_mb:.2f} MB)")
    return output_zip

if __name__ == "__main__":
    create_package()
