"""Minimal Laya-style HTTP server for SnapJudge (stdlib only, no fastapi needed).
POST /v1/systemone  {state, questions} -> {answers, routing, usage}
Run: python3 serve.py --model . --port 8000
Test: curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{"state":{"game":"snake","head":[5,5],"body":[[5,5]],"food":[7,5],"grid":[10,10]},"questions":{"direction":{"type":"choice","instructions":"Choose direction","criteria":{"up":"up","down":"down","left":"left","right":"right"}}}}'
"""
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROUTER = None

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path in ("/", "/health"):
            return self._send(200, {"ok": True, "model": "snapjudge"})
        return self._send(404, {"error": "use POST /v1/systemone"})
    def do_POST(self):
        if self.path not in ("/v1/systemone", "/predict"):
            return self._send(404, {"error": "unknown path; use POST /v1/systemone"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"invalid JSON: {e}"})
        state, questions = req.get("state"), req.get("questions")
        if state is None or questions is None:
            return self._send(422, {"error": "need {state, questions}"})
        try:
            res = ROUTER.predict(state, questions, model=req.get("model"))
        except ValueError as e:
            return self._send(422, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": repr(e)})
        return self._send(200, res)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from snapjudge.agent import load
    from snapjudge.router import GameRouter
    ROUTER = GameRouter(agent=load(a.model, device=a.device))
    # warmup
    from snapjudge.data_gen import questions_for
    ROUTER.predict({"game":"snake","head":[5,5],"body":[[5,5]],"food":[7,5],"grid":[10,10]}, questions_for("snake"))
    print(f"snapjudge serving {a.model} on 0.0.0.0:{a.port} -> POST /v1/systemone", flush=True)
    HTTPServer(("0.0.0.0", a.port), H).serve_forever()
