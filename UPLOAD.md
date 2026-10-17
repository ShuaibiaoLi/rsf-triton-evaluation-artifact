# GitHub upload checklist

## Author gates before upload

1. Confirm that the author names, affiliation, email, and acknowledgement text in the manuscript may be public.
2. Confirm that no third-party material has been added after the frozen build.
3. Run the two verification commands below; both must pass.

```bash
python3 verify_artifact.py
sha256sum -c SHA256SUMS
```

If any file changes after the frozen build, regenerate the package from the author workspace; do not hand-edit `SHA256SUMS` while leaving `artifact_metadata.json` stale.

## Create and upload the repository

### Method A: GitHub CLI

Use this method only when `gh --version` succeeds.

```bash
git init
git branch -M main
git add .
git commit -m "Release RSF Triton/HIP evaluation artifact v1"
gh auth login
gh repo create OWNER/rsf-triton-evaluation-artifact --public --source=. --remote=origin --push
git tag -a v1.0.0 -m "Frozen paper artifact v1.0.0"
git push origin v1.0.0
gh release create v1.0.0 --title "RSF Triton/HIP evaluation artifact v1.0.0" --notes "Frozen evidence package for registered-space mapping, timing semantics, and reversible RSF attribution."
```

### Method B: standard Git

Create an empty public repository named `rsf-triton-evaluation-artifact` in the GitHub web interface without adding a README or license, then run:

```bash
git init
git branch -M main
git add .
git commit -m "Release RSF Triton/HIP evaluation artifact v1"
git remote add origin https://github.com/OWNER/rsf-triton-evaluation-artifact.git
git push -u origin main
git tag -a v1.0.0 -m "Frozen paper artifact v1.0.0"
git push origin v1.0.0
```

Create the GitHub release for tag `v1.0.0` in the web interface. Replace `OWNER` before running either method. Do not claim a DOI until a Zenodo or equivalent deposit has actually minted one.
