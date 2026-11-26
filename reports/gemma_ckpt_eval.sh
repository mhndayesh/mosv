# Pick the best Gemma checkpoint on a 140-prompt dev set, then run it on the locked v6 test. Windows only, both cards.
set -e
cd /c/projects/llm_steering_rl
TR=/c/Users/mhnda/.venvs/ai-training-rocm-windows/Scripts/python.exe
GT=/c/Users/mhnda/.venvs/gguf-tools/Scripts/python.exe
S='{"temperature":1.0,"top_k":64,"top_p":0.95,"min_p":0.0,"typical_p":1.0,"repeat_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0,"top_n_sigma":-1,"xtc_probability":0.0,"dry_multiplier":0.0}'
REC="--max-new 2048 --slot-ctx 3072 --slots 13,16 --kv q8_0 --thinking off --sampling $S --seed 6"
echo "build GGUFs $(date +%T)"
HIP_VISIBLE_DEVICES=-1 $TR - <<'PY'
import sys, torch
sys.path.insert(0, "C:/llm/tools/llama.cpp/gguf-py"); sys.path.insert(0, ".")
from pathlib import Path
from moe import gemma_host
from v6.steer import GemmaSteer
from v6.train import CLASSES, LAYERS, BASE_GGUF
for u in (25, 50, 75, 100, 125):
    s = torch.load(f"checkpoints/v6/gemma_full/steer_{u:04d}.pt", weights_only=False)
    st = GemmaSteer(CLASSES, 3840, LAYERS, "cpu"); st.load(s["steer"])
    out = Path(f"C:/llm/models/gemma4-12b/v6/gemma_upd{u:03d}.gguf")
    gemma_host.build(BASE_GGUF, out, st.gguf_layers(), CLASSES, st.ff); print("built", out.name, flush=True)
PY
for tag in base:C:/llm/models/gemma4-12b/gemma-4-12b-it-Q4_K_M.gguf upd025:C:/llm/models/gemma4-12b/v6/gemma_upd025.gguf upd050:C:/llm/models/gemma4-12b/v6/gemma_upd050.gguf upd075:C:/llm/models/gemma4-12b/v6/gemma_upd075.gguf upd100:C:/llm/models/gemma4-12b/v6/gemma_upd100.gguf upd125:C:/llm/models/gemma4-12b/v6/gemma_upd125.gguf; do
  name=${tag%%:*}; model=${tag#*:}
  echo "dev answers $name $(date +%T)"
  $GT -u reports/answer_v5.py --topic v6dev --model "$model" --tag "dev_$name" $REC | tail -1
done
echo "score dev $(date +%T)"
for name in base upd025 upd050 upd075 upd100 upd125; do
  $TR reports/score_v5.py --topic v6dev --tags dev_$name | $GT -c "import sys,json; r=json.loads(sys.stdin.read()); a=r['dev_$name']['all']; print('DEVSCORE $name', a['overall'], a['tokens'], a['scored'])"
done
echo "eval done $(date +%T)"
