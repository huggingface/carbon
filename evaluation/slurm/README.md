# Evaluation SLURM scripts

SLURM scripts for running the five core DNA evals — Sequence Recovery (SR), TATA Perturbation (v1/v2), Synonymous Codon Substitution (v1/v2) — against local or HF Hub Carbon checkpoints.

## Prefix modes

All per-task sbatch scripts take a `prefix_mode` arg that controls how DNA sequences are wrapped before being fed to the model:

| Mode | Flag on the python evaluator | When to use |
|---|---|---|
| `dna_tags` | `--use_dna_tags` | Models with the **hybrid tokenizer** (wraps sequences in `<dna>…</dna>`) |
| `no_prefix` | `--no_prefix` | Raw **6-mer** / pure-DNA tokenizer models (avoids adding bos) |
| `default` | *(nothing)* | Default tokenizer behavior (Generator adds `<s>`) |

---

## Per-task sbatch scripts

### `run_seq_recovery_eval.sbatch` — Sequence Recovery

```
sbatch run_seq_recovery_eval.sbatch <model> <prefix_mode> <short_name> <step> [revision] [max_seq_len] [batch_size]
```

- `<model>` — HF repo id (e.g. `hf-carbon/carbon-3B-600B-dna-generv2`) or local HF-format dir.
- `<short_name>` — prefix for output filename (e.g. `3B-generv2`). Output is `{short_name}_{step}_bfloat16.json`.
- `<step>` — integer step (used only in filename).
- `[revision]` — optional HF branch (`step-286000`). Omit for local dirs or `main`.
- Defaults: `max_seq_len=6144`, `batch_size=64`.

**Example (hybrid tokenizer, HF hub, step 286k):**
```bash
sbatch run_seq_recovery_eval.sbatch \
    hf-carbon/carbon-3B-600B-dna-generv2 dna_tags 3B-generv2 286000 step-286000
```

### `run_tata_perturbation_eval.sbatch` / `run_tata_v2_eval.sbatch`

```
sbatch run_tata_perturbation_eval.sbatch <model> <prefix_mode> <model_name> [revision]
sbatch run_tata_v2_eval.sbatch           <model> <prefix_mode> <model_name> [revision]
```

- `<model_name>` is used directly as the output filename prefix — typically `{short_name}_{step}`, e.g. `3B-generv2_286000`.
- v1 dataset split = `tata_perturbed`, v2 = `tata_perturbed_v2`.

**Example:**
```bash
sbatch run_tata_perturbation_eval.sbatch \
    hf-carbon/carbon-3B-600B-dna-generv2 dna_tags 3B-generv2_286000 step-286000
```

### `run_synonymous_codon_eval.sbatch` / `run_synonymous_codon_v2_eval.sbatch`

Identical signature to the TATA scripts:

```
sbatch run_synonymous_codon_eval.sbatch    <model> <prefix_mode> <model_name> [revision]
sbatch run_synonymous_codon_v2_eval.sbatch <model> <prefix_mode> <model_name> [revision]
```
