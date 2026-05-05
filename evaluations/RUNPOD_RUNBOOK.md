# RunPod runbook: 3-pod parallel v2 multi-game eval (model-parallel)

End-to-end recipe for completing the v2 in-domain eval across three
RunPod pods, **split by model**. The original v2 run already produced
`<name>.json` files covering samples 0..9 (5 scenarios x 2 roles, 200
games per model), so each pod only fills in the **missing samples 10..27**
(9 scenarios x 2 roles, 360 games per model). Each pod loads ~3 of the
8 models once and evaluates each on those 18 missing samples.

Why this design:
- Each model is loaded once per pod (not 8 times) -> ~25 min saved per pod.
- Each pod only downloads the adapters it needs.
- Each pod skips the 200 games per model already done in the original v2 run.
- Per-model failure is localized (a dead pod loses 2-3 models' slice, not
  1/3 of every model's data).
- A single local merge step combines the 8 existing `<name>.json` files
  (samples 0..9) with the 8 new `<name>_samples10-27.json` slice files
  (samples 10..27) into 8 final `<name>_full14.json` files (samples 0..27).

---

## 0. What you need before starting

| key | purpose | how to get |
|---|---|---|
| OpenAI API key | GPT-4o-mini judges every dialogue outcome | platform.openai.com -> API keys |
| HuggingFace read token | downloads LoRA adapters from `migub/*` repos | huggingface.co/settings/tokens |
| RunPod account | rents the 3 GPU pods | runpod.io |

W&B is not needed (it's only for training).

---

## 1. Spin up each pod (do for pods 1, 2, 3)

In the RunPod web UI:

1. **Deploy** -> **GPU Pod**.
2. Pick a GPU. The eval uses HuggingFace `model.generate()` with 4-bit
   bitsandbytes (NF4 + double quant), no vLLM. Practical VRAM peak is
   ~12 GB for Qwen3-14B at the eval's context length (max ~2000 tokens).
   Recommendations:
     - **RTX 4090 24GB** (~$0.69/hr) -- cheapest viable.
     - **RTX A6000 48GB** (~$0.79/hr) -- best value, lots of headroom.
     - A100 / H100 only if you want maximum speed (overkill for eval).
   Avoid <16 GB cards. CUDA 12.x required.
3. **Container template**: any **PyTorch >= 2.1** template with **CUDA 12**.
   The repo's setup.sh pip-upgrades to current versions of:
     - torch >= 2.1
     - transformers >= 4.51 (Qwen3 architecture support)
     - bitsandbytes >= 0.43
     - peft >= 0.10
4. **Disk**: at least 60 GB container disk. HF cache for Qwen3-14B base
   in fp16 is ~28 GB; LoRA adapters are tiny.
5. **System RAM**: at least 32 GB. Most RunPod templates default to 50+ GB.
6. **Expose SSH** (optional; only needed if you'll use scp/rsync to pull
   results out -- you can also push to GitHub or HuggingFace instead).
7. Deploy.

Do this **three times**, one pod per group of models. Label them GPU1,
GPU2, GPU3 in the RunPod console so you don't lose track.

Cost estimate (per pod 1080 games for GPUs 1+2, 720 games for GPU 3, ~30-90s/game):
- 3x RTX 4090 ~$0.69/hr x ~5h = ~$10 total
- 3x RTX A6000 ~$0.79/hr x ~3.5h = ~$8 total
- 3x A100 80GB ~$1.89/hr x ~2.5h = ~$15 total

Total games across pods is 2880 (1080 + 1080 + 720), vs the 4480 a from-scratch
full eval would need. Skipping the already-evaluated samples 0..9 saves ~36% of
the GPU-hours.

---

## 2. Per-pod setup (run on EACH pod; same steps)

Open the pod's web terminal (or SSH in). All paths assume `/workspace`.

```bash
cd /workspace
git clone https://github.com/migub/multiturn-llm-training.git
cd multiturn-llm-training

# 2a. Pass your secrets in as env vars before the setup script.
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."

# 2b. Install deps + write secrets.json + HF login. Idempotent.
bash evaluations/runpod_setup.sh

# 2c. Download ONLY the adapters this pod needs.
bash evaluations/download_v2_checkpoints_gpu1.sh   # on pod 1
# OR
bash evaluations/download_v2_checkpoints_gpu2.sh   # on pod 2
# OR
bash evaluations/download_v2_checkpoints_gpu3.sh   # on pod 3
```

Verify (you should see 2-3 lines depending on the pod):

```bash
ls -la output/*/checkpoint-*/adapter_model.safetensors
```

Smoke-test the OpenAI key (costs a fraction of a cent):

```bash
python -c "from evaluator.openai_model import OpenAIModel; m = OpenAIModel(); \
  print(m._generate({'messages': [{'role':'user','content':'say ok'}]}))"
```

---

## 3. Run the assigned models (one model group per pod)

Run inside tmux so disconnects don't kill the job:

```bash
tmux new -s eval
cd /workspace/multiturn-llm-training

# Pod 1:
bash evaluations/run_v2_full_gpu1.sh

# Pod 2:
bash evaluations/run_v2_full_gpu2.sh

# Pod 3:
bash evaluations/run_v2_full_gpu3.sh
```

Detach: `Ctrl+B` then `D`. Reattach: `tmux attach -t eval`.

Model assignment (each model evaluated only on the missing samples 10..27):

| pod | models | games per pod |
|---|---|---|
| GPU 1 | base_model, grpo_self_only_560, grpo_fair_only_560 | 3 x 360 = 1080 |
| GPU 2 | grpo_all_equal_560, grpo_self_fair_equal_560, lagrpo_self_only_760 | 3 x 360 = 1080 |
| GPU 3 | lagrpo_all_equal_760, lagrpo_fair_only_760 | 2 x 360 = 720 |

GPU 3 has 2 models (vs 3 on the others) because LA-GRPO fair-only's
dialogues run long, so this pod's lighter game count gives a buffer.

What each pod produces (in `evaluations/results/negotiation/v2/`):

| pod | output files |
|---|---|
| GPU 1 | `base_model_samples10-27.json`, `grpo_self_only_560_samples10-27.json`, `grpo_fair_only_560_samples10-27.json` |
| GPU 2 | `grpo_all_equal_560_samples10-27.json`, `grpo_self_fair_equal_560_samples10-27.json`, `lagrpo_self_only_760_samples10-27.json` |
| GPU 3 | `lagrpo_all_equal_760_samples10-27.json`, `lagrpo_fair_only_760_samples10-27.json` |

Each slice file contains **18 sample cells x 20 reps = 360 games** covering
samples 10..27 only. The local merge step (section 5 below) pairs each
slice with its existing `<name>.json` (samples 0..9) to produce the final
`<name>_full14.json` (28 cells x 20 reps = 560 games).

The shell scripts are `set -e` and skip already-completed files
(`<name>_samples10-27.json` is checked before each run). If a pod restarts or
you interrupt mid-eval, just re-run the script and it picks up at the next
unfinished model. Mid-model interruption requires re-running that model from
scratch (the runner doesn't checkpoint within a model).

---

## 4. Pull all `_samples10-27.json` files back to your local machine

You need 8 slice files total (one per model in the panel) collected in your
local v2 directory, alongside the 8 existing `<name>.json` files (samples 0..9):

```
C:/Users/lfran/icml_paper/multiturn-llm-training/evaluations/results/negotiation/v2/
```

### Option A: rsync (recommended if SSH is exposed)

For each pod, get the SSH endpoint from the RunPod console
(**Connect -> SSH over exposed TCP**), then on your local machine:

```bash
LOCAL_DIR="/c/Users/lfran/icml_paper/multiturn-llm-training/evaluations/results/negotiation/v2"

# Pod 1
rsync -avz -e "ssh -p <port>" \
    root@<pod1-ip>:/workspace/multiturn-llm-training/evaluations/results/negotiation/v2/*_samples10-27.json \
    "$LOCAL_DIR/"

# Pod 2 (same pattern)
# Pod 3 (same pattern)
```

### Option B: push to GitHub from each pod

On each pod (after setting `GH_TOKEN` env var with `repo` scope):

```bash
git remote set-url origin https://${GH_TOKEN}@github.com/migub/multiturn-llm-training.git
git config user.email "fluca182@gmail.com"
git config user.name  "Luca Franceschetti (RunPod)"

# Per-pod branch to avoid races between pods.
git checkout -B "eval/v2-pod${POD_NUM}"   # POD_NUM = 1, 2, or 3
git add evaluations/results/negotiation/v2/*_samples10-27.json
git commit -m "v2 samples10-27 slice: pod ${POD_NUM} models"
git push -u origin "eval/v2-pod${POD_NUM}" --force-with-lease
```

Locally:

```bash
cd C:/Users/lfran/icml_paper/multiturn-llm-training
git fetch origin eval/v2-pod1 eval/v2-pod2 eval/v2-pod3
git checkout origin/eval/v2-pod1 -- evaluations/results/negotiation/v2/*_samples10-27.json
git checkout origin/eval/v2-pod2 -- evaluations/results/negotiation/v2/*_samples10-27.json
git checkout origin/eval/v2-pod3 -- evaluations/results/negotiation/v2/*_samples10-27.json
```

(The three checkouts pull non-overlapping files because each pod produced
disjoint model results.)

### Option C: runpodctl

`runpodctl send <files>` from the pod, copy the receive command to your
local machine. Web-based, no SSH config needed.

---

## 5. Merge locally and verify

After step 4 the local v2 directory contains, for each model:
- `<name>.json` (samples 0..9, from the original v2 run)
- `<name>_samples10-27.json` (samples 10..27, from the pod that owned this model)

Run:

```bash
cd C:/Users/lfran/icml_paper/multiturn-llm-training

# Auto-merge: every <name>.json + its <name>_samples10-27.json -> <name>_full14.json
python evaluations/merge_eval_slices.py --batch evaluations/results/negotiation/v2

# Verify each merged file covers all 28 sample cells, 7 archetypes, 20 reps,
# n_games == 560.
python evaluations/verify_full14.py evaluations/results/negotiation/v2
```

`verify_full14.py` exits 0 with `ALL OK` on success. If it prints
`INCOMPLETE`, it lists which cells or archetypes are missing per file so
you know which model to re-run.

You'll have 8 new files: `*_full14.json`, one per model in the panel.
Those are the canonical artifacts for the paper's headline numbers.

---

## 6. Shut down the pods

In the RunPod console: **My Pods** -> select pod -> **Terminate** for all
three. RunPod bills per minute while pods exist, so terminate when done.

---

## Troubleshooting

**"OPENAI_API_KEY not found"** when starting a game.
Check `secrets.json` exists in repo root and contains valid JSON of the
form `{"openai": {"api_key": "sk-..."}}`.

**`hf download` fails with 401.**
The `migub/*` repos may require auth. Set `HF_TOKEN` and re-run
`runpod_setup.sh`, or `huggingface-cli login` and paste the token.

**Out-of-memory on smaller GPUs.**
Qwen3-14B in 4-bit needs ~12 GB peak at this context. 24 GB cards are
fine. If you OOM on a 16 GB card, switch to 24 GB or larger.

**Mid-model crash / pod restart.**
Re-run the same shell script. It checks for existing `_full14.json` files
and skips completed models, so it picks up at the next unfinished model.
Mid-model crashes lose that model's progress -- it'll restart from rep 0.

**OpenAI rate limits.**
Standard tier handles ~30 calls/sec. If you hit a 429, the OpenAI client
retries automatically. Sustained 429s mean you should bump your tier or
stagger the pods slightly in time.

**One pod is much slower than the others.**
LA-GRPO fair-only produces longer dialogues than the other models (fewer
early closures), so its per-game wall time is ~30% longer. The 3/3/2
model split puts that model on GPU 3 alone with a lighter load, but it
may still finish last. Don't merge until all three are done; the verifier
will tell you which models are missing.

---

## File index

In the repo (already in place):
- `evaluations/run_negotiation_eval.py` -- patched to support `--sample-start-idx`
- `evaluations/runpod_setup.sh` -- one-shot per-pod setup
- `evaluations/download_v2_checkpoints_gpu1.sh` -- downloads pod 1's adapters
- `evaluations/download_v2_checkpoints_gpu2.sh` -- downloads pod 2's adapters
- `evaluations/download_v2_checkpoints_gpu3.sh` -- downloads pod 3's adapters
- `evaluations/run_v2_full_gpu1.sh` -- pod 1's 3 models, samples 10..27
- `evaluations/run_v2_full_gpu2.sh` -- pod 2's 3 models, samples 10..27
- `evaluations/run_v2_full_gpu3.sh` -- pod 3's 2 models, samples 10..27
- `evaluations/merge_eval_slices.py` -- combines `<name>.json` (samples 0..9)
  with `<name>_samples10-27.json` into `<name>_full14.json`
- `evaluations/verify_full14.py` -- structural check on `_full14.json` files
