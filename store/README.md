# Publishing Gorget

Checklist for the maintainer. Nothing here runs automatically except the release workflow.

## Once

1. Create the GitHub repository `Perruer/gorget` and push `main`.
2. Settings → General → Social preview: upload `store/social-preview.png`.
3. Settings → General → Features: tick **Sponsorships** (FUNDING.yml needs it).
4. Settings → Pages: deploy from the `gh-pages` branch (the Docs workflow creates it).
5. PyPI: create the project `gorget` with a trusted publisher
   (owner `Perruer`, repository `gorget`, workflow `release.yml`, environment `pypi`),
   then set the repository variable `PYPI_PUBLISH=true`.
6. Optional: a read-only Hugging Face token as the `HF_TOKEN` secret for CI.

## Each release

1. Update `version` in `pyproject.toml` and `gorget_api/app/version.py`, date the changelog.
2. Write `store/github-release-v<version>.md` (used as release notes).
3. Tag `v<version>` and push the tag. The Release workflow builds the wheel and sdist,
   drafts the GitHub release with them and `SHA256SUMS`, publishes to PyPI (when enabled)
   and pushes `ghcr.io/perruer/gorget-api:<version>` and `:<version>-cuda`.
4. Check the draft and publish it.
