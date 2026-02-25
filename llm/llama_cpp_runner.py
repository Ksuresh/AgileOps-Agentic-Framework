from __future__ import annotations
import subprocess

def run_llama(llama_bin: str, gguf_path: str, system_prompt: str, user_prompt: str,
             temperature: float, top_p: float, max_tokens: int) -> str:
    cmd = [
        llama_bin,
        "-m", gguf_path,
        "--temp", str(temperature),
        "--top-p", str(top_p),
        "-n", str(max_tokens),
        "--prompt", f"<s>[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n{user_prompt}"
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return out
