from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".json", ".md", ".yaml", ".yml", ".toml", ".txt", ".css", ".html"}
EXCLUDED_EVIDENCE = {
    Path(__file__).resolve(),
    ROOT / "results" / "face" / "HYPERPARAMETER_AUDIT.md",
}


def text_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS and "node_modules" not in path.parts and "dist" not in path.parts:
            yield path


def main():
    findings = []
    required = [
        "src/face", "src/speech", "src/numerical", "src/fusion", "src/api",
        "frontend", "configs", "scripts", "results/face", "results/speech",
        "results/numerical", "results/fusion", "docs", "tests",
        "README.md", "results/FINAL_RESULTS.md",
    ]
    for relative in required:
        if not (ROOT / relative).exists():
            findings.append(f"missing required path: {relative}")
    secret_patterns = [
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ]
    forbidden = [
        "Res" + "EmoteNet",
        "0." + "5679",
        "C:\\Users\\" + "Preetha",
        "new " + "implementation",
        "training_" + "in_progress",
    ]
    for path in text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in secret_patterns:
            if pattern.search(text): findings.append(f"secret-like value: {path.relative_to(ROOT)}")
        if path not in EXCLUDED_EVIDENCE:
            for token in forbidden:
                if token in text: findings.append(f"stale token {token!r}: {path.relative_to(ROOT)}")
    lfs_patterns = {".pt", ".pth", ".joblib"}
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or ".cache" in path.parts or "node_modules" in path.parts or "dist" in path.parts:
            continue
        is_selected_lfs_model = path.parent == ROOT / "models" and path.suffix.lower() in lfs_patterns
        if path.is_file() and path.stat().st_size > 95 * 1024 * 1024 and not is_selected_lfs_model:
            findings.append(f"GitHub-large file: {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    link_pattern = re.compile(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)")
    for path in ROOT.rglob("*.md"):
        if "node_modules" in path.parts or "dist" in path.parts:
            continue
        for target in link_pattern.findall(path.read_text(encoding="utf-8", errors="ignore")):
            if not target.startswith(("http://", "https://", "mailto:")) and not (path.parent / target).resolve().exists():
                findings.append(f"broken local link: {path.relative_to(ROOT)} -> {target}")
    if findings:
        print("\n".join(findings)); return 1
    print("Submission scan passed: no secrets, stale model names/metrics, private absolute paths, or untracked large files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
