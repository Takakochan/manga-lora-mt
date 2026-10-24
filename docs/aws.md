# Running the GPU experiments on AWS

[日本語版](aws.ja.md)

The Qwen3 training and inference in `scripts/lora.py` need an NVIDIA GPU. This guide runs them on
one AWS EC2 GPU instance, then brings the predictions back for `scripts/eval.py`. The Kimi-K3 run
and the evaluation don't need a GPU and can run on any PC.

## 1. Before you start (one time)

1. **Request GPU quota.** New AWS accounts can't launch GPU instances until the quota is raised.
   In the AWS console, open **Service Quotas → Amazon EC2 → Running On-Demand G and VT
   instances** and request **8** (the unit is vCPUs). Quotas are per region, so request it in the
   region you'll use, e.g. `us-east-1`. Approval can take from a few hours to a couple of days.
2. **Check your credits.** In **Billing → Credits**, confirm the credits cover Amazon EC2.
3. **Create a key pair.** In **EC2 → Key pairs**, create one (`.pem`) and save it on the PC you'll
   connect from. Then run `chmod 400 your-key.pem`.

## 2. Launch the instance

In **EC2 → Launch instance**:

| setting | value |
|---|---|
| AMI | an Ubuntu 22.04 **Deep Learning AMI** with PyTorch (search "Deep Learning AMI"); it includes the NVIDIA driver |
| instance type | `g6.2xlarge` (L4, 24 GB GPU, 32 GB RAM); `g5.2xlarge` (A10G) also works |
| key pair | the one from step 1 |
| network | allow SSH from **My IP** only |
| storage | 100 GB gp3 |

If the launch fails with `VcpuLimitExceeded`, the quota isn't approved yet. If it fails with
`InsufficientInstanceCapacity`, try another availability zone or region.

## 3. Connect and set up

```
ssh -i your-key.pem ubuntu@<public DNS of the instance>
tmux new -s lora
```

Work inside `tmux` so the jobs keep running if SSH drops. Reattach with `tmux attach -t lora`.

Clone the repository. If it's private, the instance needs access: either run `gh auth login`, or
clone over HTTPS with a GitHub personal access token as the password.

```
git clone https://github.com/Takakochan/manga-lora-mt.git
cd manga-lora-mt
scripts/aws_setup.sh
```

`scripts/aws_setup.sh` checks the GPU, installs uv, syncs the `finetune` extra, downloads the data
and runs a short smoke-test training run. It ends with `smoke test passed`.

## 4. Run the experiment

```
uv run scripts/lora.py predict                               # base Qwen3-1.7B, all 5 folds
uv run scripts/lora.py train                                 # 5 adapters
uv run scripts/lora.py predict --adapters Qwen3-1.7B-lora    # Qwen3-1.7B + LoRA
```

If training runs out of GPU memory, add `--batch-size 1 --grad-accum 8` to `train`.

The ablations from the README's Plan run the same way. For example:

```
uv run scripts/lora.py train --rank 8 --run Qwen3-1.7B-lora-r8
uv run scripts/lora.py predict --adapters Qwen3-1.7B-lora-r8
```

## 5. Bring the results back

From the PC you'll evaluate on, inside the repository:

```
scp -i your-key.pem -r ubuntu@<public DNS>:manga-lora-mt/out/mt/preds/ out/mt/
scp -i your-key.pem -r ubuntu@<public DNS>:manga-lora-mt/out/mt/adapters/ out/mt/
```

The adapters are only needed if you want to reuse them later (for example, serving with vLLM).

## 6. Evaluate

`eval.py` needs every run's predictions in `out/mt/preds/` on the same PC. The Kimi-K3
predictions aren't in git, so either copy `out/mt/preds/Kimi-K3/` from the PC that ran it, or
regenerate them with `NEBIUS_API_TOKEN` in `.env`:

```
uv run scripts/data.py
uv run scripts/predict_api.py
```

Then:

```
uv run scripts/eval.py Qwen3-1.7B,Qwen3-1.7B-lora,Kimi-K3
```

## 7. Stop paying

- **Stop** the instance when you pause. You aren't charged for the GPU while it's stopped, but you
  still pay for the 100 GB disk.
- **Terminate** it when you're done and have copied the results back. Terminating deletes the disk
  and everything on it.
- Check **Billing → Bills** after the first session to confirm the charges go to your credits.
