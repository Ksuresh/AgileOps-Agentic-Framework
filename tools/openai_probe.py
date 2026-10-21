"""One tiny API call; writes status (no secrets) to tools/openai_probe.json."""
import json, os, time
from openai import OpenAI
out = {"model": os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")}
t0 = time.time()
try:
    r = OpenAI(timeout=60, max_retries=0).responses.create(model=out["model"], reasoning={"effort": "none"}, input="Reply with OK.")
    out.update(ok=True, text=(r.output_text or "")[:20])
except Exception as e:
    out.update(ok=False, error_type=type(e).__name__, error=str(e)[:400])
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open("tools/openai_probe.json", "w"), indent=1)
print(out)
