# Environment

```text
OPEN_DATA_COMMIT=a0c99668c3690f9051f0af0161c8aea19b17d59f
MODEL_CODE_COMMIT=6a398b13df94823b554447c8d66e9bcfc8813b6d
WORK_BRANCH=cc-event-transfer-confirmation-20260903
PYTHON_VERSION=3.11.6
TORCH_VERSION=2.11.0
DEVICE=cpu
PLATFORM=macOS-13.7.3-arm64-arm-64bit
```

`MODEL_CODE_COMMIT` is this repository, **not** the data-challenge repository named in
the task prompt. That repository is excluded by `FIREWALL.md`; it was not cloned, read
or referenced, and the public two-rate implementation used here has always lived in
this repository. The exploratory branch `gpt-pretest-20260903` (`9a3409a5`) was listed
only; none of its formulas or numbers entered this work.

Data digests: `panel_digest=76a73ed794af`,
`channel_digest=dec964873cb2`,
`event_split_digest=aea2acb10037`,
`county_split_digest=f5a428dfa590`.
Data checksums: 60/60 files verified against `data/SHA256SUMS.txt`.
