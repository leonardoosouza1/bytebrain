#!/usr/bin/env python3
"""CÓRTEX SERVER — servidor HTTP mínimo (stdlib, ZERO deps além do llama_cpp) que serve o Qwen2.5-7B GGUF
como córtex do IARA-BRAIN. Isola o llama_cpp (HIP) do torch da memória (não coexistem no mesmo processo).
Endpoint OpenAI: POST /v1/chat/completions ; GET /v1/models. Rode isolado:
  HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 research/cortex_server.py"""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from llama_cpp import Llama
MODEL = "/home/leonardo/projects/LLM/llm-lab/models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
print("carregando Qwen2.5-7B (córtex)...", flush=True)
llm = Llama(model_path=MODEL, n_gpu_layers=-1, n_ctx=4096, chat_format="chatml", verbose=False)
print("CÓRTEX ONLINE :8080", flush=True)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/v1/models"): self._send({"data": [{"id": "qwen2.5-7b", "object": "model"}]})
        else: self._send({"ok": True})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        try:
            out = llm.create_chat_completion(messages=req.get("messages", []),
                    max_tokens=req.get("max_tokens", 350), temperature=req.get("temperature", 0.5))
            self._send(out)
        except Exception as e:
            self._send({"error": str(e)}, 500)
if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8080), H).serve_forever()
