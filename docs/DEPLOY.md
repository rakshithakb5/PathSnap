# Two-laptop deployment (Phase 6)

```
 LAPTOP 2 (UI / evaluator view)                         LAPTOP 1 (Model + API)
 ┌──────────────────────────┐   POST /predict {trip}    ┌────────────────────────────────────────────┐
 │ browser: ui/index.html   │ ───────────────────────►  │ FastAPI (uvicorn)  api/main.py             │
 │  - lists the 10 trips    │                           │  1. load trip files (NO labels)            │
 │  - sends trip id         │ ◄───────────────────────  │  2. candidates + features                  │
 │  - draws predicted route │  route + GPS + metrics    │  3. LightGBM probabilities                 │
 │  - shows accuracy table  │                           │  4. HMM / Viterbi route                    │
 └──────────────────────────┘                           │  5. score vs reference (server side only)  │
        shows NO ground truth                           │  console log of every request  ← show this │
                                                        └────────────────────────────────────────────┘
```
**Rule:** ground truth stays on Laptop 1. The UI only sends a trip id and displays what comes back.

## Laptop 1 – Model and API (the "backend")
Needs: the repo, `pip install -r requirements.txt`, `artifacts/final/{model.pkl,hmm_config.json}` (created by Phase 5; commit them),
and the evaluator-trip folders (the organisers give you the 10 trips on the day; same layout as `student_test`).

1. Join the shared Wi-Fi / hotspot. Find your IP:  `ipconfig`  → the IPv4 address, e.g. `192.168.1.23`.
2. Allow the port through Windows Firewall (run once, as Administrator PowerShell):
   `New-NetFirewallRule -DisplayName "PathSnap" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
3. Point the API at the evaluator trips and start it (CMD, from the repo root):
   ```
   set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_test
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   `--host 0.0.0.0` is essential: the default (127.0.0.1) only accepts connections from the same machine.
4. Check locally: open `http://localhost:8000/docs` (auto-generated API page) or `curl http://localhost:8000/health`.
5. Keep this console visible. Every request prints `REQUEST ... / RESPONSE ... Top-1 ...` lines. **These are the API logs
   the evaluator must see, and the screenshots the report needs ("successful API responses on the CMD of the receiving laptop").**

## Laptop 2 – UI (the "frontend")
Needs: only the `ui/index.html` file and a browser. No Python, no model, no data.
1. Same network as Laptop 1. Test reachability: open `http://<LAPTOP1-IP>:8000/health` in the browser – you should see `"status":"ok"`.
2. Open `ui/index.html` (double-click). Type `http://<LAPTOP1-IP>:8000` in the API address box → **Connect & load trips**.
3. Pick a trip → **Run prediction** → the predicted route is drawn and the metrics table appears.
4. The UI draws on a plain canvas (no map tiles), so it works even with no internet.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| UI says "Cannot reach API" | wrong IP, different Wi-Fi, firewall rule missing, or API started without `--host 0.0.0.0` |
| `/trips` is empty | `PATHSNAP_EVAL_DIR` not set / wrong folder (must contain one sub-folder per trip) |
| "falling back to lgbm.pkl" warning | `artifacts/final/model.pkl` missing – run Phase 5 (or Phase 4 then 5) and commit `artifacts/final/` |
| Slow first request | normal (library warm-up); a trip takes ~1–2 s afterwards |
| Browser blocks the request (mixed content) | open the UI as a local file or via `http://`, not a https page |

## Pre-demo checklist
- [ ] Both laptops charged, same network, both IPs noted
- [ ] `curl http://<IP>:8000/health` works from Laptop 2
- [ ] All 10 trips run without error (try them all once before the evaluator arrives)
- [ ] API console visible; screenshots taken (UI + API log)
- [ ] The code in GitHub is exactly the code running (commit before the demo, no secrets in the repo)
