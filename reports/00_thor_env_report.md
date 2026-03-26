# NVIDIA Thor Development Environment Report

---

## 1. GPU -- nvidia-smi

```
NVIDIA-SMI 580.00 | Driver Version: 580.00 | CUDA Version: 13.0

GPU 0: NVIDIA Thor
  Architecture:       Blackwell
  Compute Capability: 11.0
  Persistence Mode:   Enabled
  Addressing Mode:    ATS (unified/shared memory -- Tegra SoC, no dedicated VRAM)
  Temperature:        34C
  Memory Total:       N/A (unified memory, shared with CPU -- no dedicated VRAM)
  Current Processes:  Xorg (48 MiB), gnome-shell (25 MiB)
```

---

## 2. CUDA Compiler -- nvcc

| Item | Value |
|------|-------|
| Path | `/usr/local/cuda-13.0/bin/nvcc` |
| In default PATH | No -- must add manually |
| Version | CUDA 13.0.48 (built Wed Jul 16 2025) |
| Full toolkit installed | Yes, via apt (`cuda-*-13-0` packages) |

To use nvcc: `export PATH=$PATH:/usr/local/cuda-13.0/bin`

---

## 3. Python Environments

| Environment | Path | Python Version |
|-------------|------|----------------|
| System Python | `/usr/bin/python3` | 3.12.3 |
| qwen3_thor venv | `/home/nvidia/qwen3_thor/venv` | 3.12 |
| smolvlm_thor | `/home/nvidia/smolvlm_thor/` | No venv (scripts/trt dirs only) |
| smolvlm_orin | `/home/nvidia/smolvlm_orin/` | Not inspected |

- **conda:** Not installed
- **uv:** Not installed
- **pip:** 24.0 at `/usr/bin/pip3`

---

## 4. PyTorch

| Location | Version | CUDA Build | CUDA Available |
|----------|---------|-----------|----------------|
| System pip | 2.9.1+**cpu** | None | False |
| qwen3_thor venv | 2.10.0+**cpu** | None | False |

**WARNING: No CUDA-enabled PyTorch installed. Both installs are CPU-only builds.**
A Tegra/ARM64 CUDA-enabled PyTorch wheel must be installed for GPU workloads.

### Other notable system pip packages

| Package | Version |
|---------|---------|
| tensorrt | 10.13.3.9 |
| tensorrt_dispatch | 10.13.3.9 |
| tensorrt_lean | 10.13.3.9 |
| torchvision | 0.24.1 (CPU) |
| numpy | 1.26.4 |
| safetensors | 0.7.0 |
| torchprofile | 0.0.4 |

---

## 5. NVIDIA Container Runtime

| Item | Status |
|------|--------|
| Docker version | 28.2.2 (linux/arm64) |
| nvidia-container-toolkit | 1.18.1-1 installed |
| libnvidia-container1 | 1.18.1-1 installed |
| Docker daemon.json | NOT found -- nvidia runtime not configured |
| User docker access | DENIED -- nvidia not in docker group |
| NVIDIA images pulled | None |

**To enable NVIDIA Docker runtime:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker nvidia
```

---

## 6. Disk Space

| Filesystem | Size | Used | Available | Use% | Mount |
|-----------|------|------|-----------|------|-------|
| /dev/nvme0n1p1 | 937G | 326G | 564G | 37% | / |
| tmpfs (shm) | 62G | 88K | 62G | <1% | /dev/shm |

---

## 7. CPU Architecture

| Item | Value |
|------|-------|
| Architecture | aarch64 (ARM64) |
| Vendor | ARM |
| CPU cores | 14 (single cluster, 1 thread/core) |
| Platform | NVIDIA Thor SoC (Tegra/Jetson) |
| JetPack | R38, Revision 4.0 (built Dec 31 2025) |

---

## 8. Package Managers

| Tool | Available | Notes |
|------|-----------|-------|
| pip | Yes | 24.0 at /usr/bin/pip3 |
| uv | No | Not installed |
| conda | No | Not installed |

---

## 9. Internet Connectivity

| Host | Reachable |
|------|-----------|
| github.com | Yes |
| pypi.org | Yes |

Note: `curl` is not installed on this system; tested via `wget`.

---

## 10. Summary

| Item | Status |
|------|--------|
| GPU detected | NVIDIA Thor (Blackwell, sm_110, compute cap 11.0) |
| CUDA 13.0 toolkit | Installed at /usr/local/cuda-13.0 |
| nvcc in PATH | No -- add /usr/local/cuda-13.0/bin to PATH |
| PyTorch with CUDA | NOT AVAILABLE -- CPU-only builds only |
| TensorRT | 10.13.3.9 installed (system pip) |
| Docker + nvidia-ctk | Installed but not configured |
| Internet | github.com and pypi.org reachable |
| Disk free | 564 GB on NVMe |

### Action Items

1. **Install CUDA PyTorch:** Get Tegra/ARM64 CUDA wheel from NVIDIA's Jetson PyTorch releases
2. **Configure nvidia Docker runtime:** `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`
3. **Add nvcc to PATH:** Append `export PATH=$PATH:/usr/local/cuda-13.0/bin` to `~/.bashrc`
4. **Add user to docker group:** `sudo usermod -aG docker nvidia`
