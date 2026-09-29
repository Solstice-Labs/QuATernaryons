import modal
import os

app = modal.App("merge-ggufs")

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "tar", "build-essential", "cmake", "git", "libcurl4-openssl-dev")
    .pip_install("huggingface-hub>=0.20.0", "requests")
    .run_commands(
        # Download prebuilt llama.cpp binaries
        "curl -fsSL https://github.com/ggml-org/llama.cpp/releases/download/b10831/llama-b10831-bin-ubuntu-x64.tar.gz -o /tmp/llama.tar.gz",
        "tar xzf /tmp/llama.tar.gz -C /usr/local/bin --strip-components=0",
        "rm /tmp/llama.tar.gz",
        "chmod +x /usr/local/bin/llama-gguf-split || true",
        # Find and symlink the binary
        "ls /usr/local/bin/llama-* 2>/dev/null || find /usr/local -name 'llama-gguf-split' 2>/dev/null",
    )
)

volume = modal.Volume.from_name("gguf-workspace", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[hf_secret],
    timeout=3600 * 8,
)
def merge():
    from huggingface_hub import hf_hub_download, HfApi
    import subprocess
    import shutil
    from pathlib import Path

    REPO = "Solstice-AI/DeepSeek-V4-Flash-Vision-Exp-UltraOptimised-GGUF"
    SRC_REPO = "unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF"

    WORK = Path("/vol/work")
    WORK.mkdir(exist_ok=True)

    api = HfApi()

    # Check what binaries we got
    print("Available llama binaries:")
    for f in Path("/usr/local/bin").glob("llama-*"):
        print(f"  {f.name}")

    # Find gguf-split or gguf合并 tool
    split_bin = None
    for name in ["llama-gguf-split", "gguf-split"]:
        p = Path(f"/usr/local/bin/{name}")
        if p.exists():
            split_bin = str(p)
            break

    if not split_bin:
        # List what we have
        result = subprocess.run(["ls", "-la", "/usr/local/bin/llama-*"], capture_output=True, text=True)
        print(f"Binaries found: {result.stdout}")
        # Try to find it in subdirectories
        result2 = subprocess.run(["find", "/usr/local", "-name", "*gguf*split*", "-type", "f"], capture_output=True, text=True)
        print(f"Search results: {result2.stdout}")
        # If tar extracted into a subdirectory
        for p in Path("/usr/local").rglob("llama-gguf-split"):
            split_bin = str(p)
            break

    if not split_bin:
        # Fallback: build from source
        print("gguf-split not found in prebuilt, building from source...")
        subprocess.run([
            "git", "clone", "--depth=1", "https://github.com/ggml-org/llama.cpp.git", "/tmp/llama-cpp"
        ], check=True)
        subprocess.run([
            "cmake", "-B", "/tmp/llama-cpp/build", "-S", "/tmp/llama-cpp",
            "-DLLAMA_CURL=OFF"
        ], check=True)
        subprocess.run([
            "cmake", "--build", "/tmp/llama-cpp/build", "-j", "--target", "llama-gguf-split"
        ], check=True)
        # Find the built binary
        for p in Path("/tmp/llama-cpp/build").rglob("llama-gguf-split"):
            split_bin = str(p)
            shutil.copy(split_bin, "/usr/local/bin/llama-gguf-split")
            split_bin = "/usr/local/bin/llama-gguf-split"
            break

    print(f"Using split binary: {split_bin}")

    merge_jobs = [
        ("UD-Q8_K_XL", 151),
        ("UD-Q2_K_XL", 90),
    ]

    for tier, expected_gb in merge_jobs:
        print(f"\n{'='*60}")
        print(f"Processing {tier} (~{expected_gb} GB)")
        print(f"{'='*60}")

        tier_dir = WORK / tier
        if tier_dir.exists():
            shutil.rmtree(tier_dir)
        tier_dir.mkdir(parents=True, exist_ok=True)

        # List all files in this tier
        print(f"Listing files in {SRC_REPO}/{tier}/...")
        files = api.list_repo_files(SRC_REPO, repo_type="model")
        shard_files = sorted([
            f for f in files
            if f.startswith(f"{tier}/") and f.endswith(".gguf")
        ])
        print(f"Found {len(shard_files)} shards")

        # Download all shards into tier dir
        for shard in shard_files:
            local = hf_hub_download(
                repo_id=SRC_REPO,
                filename=shard,
                local_dir=str(WORK),
                repo_type="model",
            )
            size_gb = os.path.getsize(local) / 1e9
            print(f"  Downloaded: {os.path.basename(local)} ({size_gb:.2f} GB)")

        # Merge shards into single file
        output_name = f"DeepSeek-V4-Flash-Vision-Exp-{tier}.gguf"
        output_path = str(WORK / output_name)

        # llama-gguf-split --merge <shard1> <output>
        first_shard = str(tier_dir / shard_files[0].split("/")[-1])
        print(f"Merging into {output_name}...")

        result = subprocess.run(
            [split_bin, "--merge", first_shard, output_path],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            print(f"Merge failed: {result.stderr}")
            # Try alternative: maybe the syntax is different
            print(f"Trying alternative merge syntax...")
            result = subprocess.run(
                [split_bin, "merge", first_shard, output_path],
                capture_output=True, text=True, timeout=3600
            )
            if result.returncode != 0:
                print(f"Also failed: {result.stderr}")
                # Fallback: just upload shards individually (no merge)
                print("Merge failed, uploading shards individually...")
                for shard in shard_files:
                    shard_local = str(WORK / shard)
                    shard_name = os.path.basename(shard)
                    api.upload_file(
                        path_or_fileobj=shard_local,
                        path_in_repo=shard_name,
                        repo_id=REPO,
                        repo_type="model",
                    )
                    print(f"  Uploaded: {shard_name}")
                continue

        merged_size = os.path.getsize(output_path) / 1e9
        print(f"✅ Merged: {output_name} ({merged_size:.2f} GB)")

        # Upload
        print(f"Uploading {output_name} to {REPO}...")
        api.upload_file(
            path_or_fileobj=output_path,
            path_in_repo=output_name,
            repo_id=REPO,
            repo_type="model",
        )
        print(f"✅ Uploaded: {output_name}")

        # Cleanup
        shutil.rmtree(tier_dir, ignore_errors=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        volume.reload()

    # Also keep mmproj-BF16 flat
    print(f"\n{'='*60}")
    print("Copying mmproj-BF16.gguf")
    print(f"{'='*60}")
    mmproj = hf_hub_download(
        repo_id=SRC_REPO,
        filename="mmproj-BF16.gguf",
        local_dir=str(WORK),
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=mmproj,
        path_in_repo="mmproj-BF16.gguf",
        repo_id=REPO,
        repo_type="model",
    )
    print("✅ Uploaded: mmproj-BF16.gguf")

    print("\n🎉 Done! Flat merged GGUFs in repo.")


@app.local_entrypoint()
def main():
    merge.remote()
