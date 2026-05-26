# pip install ollama fastapi uvicorn
# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

import asyncio
import base64
import gc
import os
import subprocess
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from ollama import AsyncClient

# =========================
# CONFIG
# =========================

MODEL_NAME = "qwen3.5:0.8b"
OLLAMA_HOST = "http://localhost:11434"

# Shared client — created once, reused across all requests
ollama_client: AsyncClient | None = None


# =========================
# HELPERS
# =========================

def build_message(prompt: str, image_b64: str | None) -> dict:
    """Build a single user message dict, optionally with a base64 image."""
    msg: dict = {"role": "user", "content": prompt}
    if image_b64:
        msg["images"] = [image_b64]
    return msg


def log(icon: str, label: str, detail: str = "") -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {icon}  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)


# =========================
# LIFESPAN (startup / shutdown)
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama_client

    t0 = time.perf_counter()
    log("🧹", "STARTUP", "Cleaning Python memory...")
    gc.collect()
    log("✅", "GC", "Python garbage collection done")

    # Linux page-cache flush (only works as root — skipped silently otherwise)
    try:
        subprocess.run(["sync"], check=True, capture_output=True)
        subprocess.run(
            ["tee", "/proc/sys/vm/drop_caches"],
            input=b"3",
            check=True,
            capture_output=True,
        )
        log("🧹", "SYSCACHE", "Linux page cache dropped (drop_caches=3)")
    except Exception as e:
        log("⚠️", "SYSCACHE SKIPPED", str(e))

    log("🔌", "CLIENT", f"Creating shared AsyncClient → {OLLAMA_HOST}")
    ollama_client = AsyncClient(host=OLLAMA_HOST)

    log("🚀", "WARMUP", f"Pre-loading model: {MODEL_NAME}")
    try:
        t_warm = time.perf_counter()
        res = await ollama_client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "warmup"}],
            think=False,
            options={"num_ctx": 128,},
        )
        elapsed = time.perf_counter() - t_warm
        log("✅", "WARMUP DONE", f"Model '{MODEL_NAME}' pinned in RAM  [{elapsed:.2f}s]")
        print(f"  [warmup response] {res.message.content[:200]}{'...' if len(res.message.content) > 200 else ''}")
    except Exception as e:
        log("⚠️", "WARMUP FAILED", str(e))

    total = time.perf_counter() - t0
    log("🟢", "SERVER READY", f"Startup completed in {total:.2f}s")
    yield

    log("🛑", "SHUTDOWN", "Server shutting down — goodbye")


# =========================
# APP
# =========================

app = FastAPI(title="Flexible Ollama API", lifespan=lifespan)


# =========================
# REQUEST LOGGING MIDDLEWARE
# =========================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    log("📥", "REQUEST", f"{request.method} {request.url.path}")
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    log("📤", "RESPONSE", f"status={response.status_code}  [{elapsed:.1f}ms]")
    return response


# =========================
# STREAM GENERATOR
# =========================

async def stream_response(prompt: str, image_b64: str | None, options: dict):
    log("🌊", "STREAM START", f"model={MODEL_NAME}  prompt_len={len(prompt)}")
    token_count = 0
    t0 = time.perf_counter()

    try:
        message = build_message(prompt, image_b64)
        stream = await ollama_client.chat(  # type: ignore[union-attr]
            model=MODEL_NAME,
            messages=[message],
            stream=True,
            think=False,
            options={**options},  # think=False belongs in options
        )

        async for chunk in stream:
            token = chunk.message and chunk.message.content
            if token:
                token_count += 1
                # Detailed per-token log (comment out in production for speed)
                print(f"  [token #{token_count:04d}] {repr(token)}")
                yield f"data: {token}\n\n"

    except Exception as e:
        log("❌", "STREAM ERROR", str(e))
        yield f"data: [ERROR]: {str(e)}\n\n"

    finally:
        elapsed = time.perf_counter() - t0
        log(
            "✅",
            "STREAM DONE",
            f"tokens={token_count}  time={elapsed:.2f}s  "
            f"speed={token_count / elapsed:.1f} tok/s" if elapsed > 0 else "tokens=0",
        )


# =========================
# MAIN ROUTE (FLEXIBLE)
# =========================

@app.post("/chat")
async def chat(
    prompt: str = Form(...),
    stream: bool = Form(True),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    top_k: int = Form(40),
    repeat_penalty: float = Form(1.5),
    max_tokens: int = Form(200),
    image: UploadFile | None = File(None),
):
    log(
        "💬",
        "CHAT",
        f"stream={stream}  temp={temperature}  top_p={top_p}  "
        f"top_k={top_k}  repeat_penalty={repeat_penalty}  max_tokens={max_tokens}",
    )
    log("📝", "PROMPT", f"[{len(prompt)} chars] {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

    # ---- IMAGE ----
    image_b64: str | None = None

    if image:
        content_type = image.content_type or ""
        if not content_type.startswith("image/"):
            log("❌", "IMAGE REJECTED", f"content_type='{content_type}'")
            raise HTTPException(status_code=400, detail="Only images allowed")

        raw_bytes = await image.read()
        image_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        log(
            "🖼️",
            "IMAGE",
            f"filename='{image.filename}'  type='{content_type}'  "
            f"size={len(raw_bytes):,} bytes  b64_len={len(image_b64):,} chars",
        )

    # ---- OPTIONS ----
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "num_predict": max_tokens,
        "num_ctx": 4096,
    }
    log("⚙️", "OPTIONS", str(options))

    # ---- STREAM MODE ----
    if stream:
        log("🌊", "MODE", "Streaming SSE response")
        return StreamingResponse(
            stream_response(prompt, image_b64, options),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",   # Prevent Nginx from buffering SSE
                "Cache-Control": "no-cache",
            },
        )

    # ---- FULL RESPONSE MODE ----
    log("📦", "MODE", "Buffered (non-streaming) response")
    t0 = time.perf_counter()

    try:
        message = build_message(prompt, image_b64)
        response = await ollama_client.chat(  # type: ignore[union-attr]
            model=MODEL_NAME,
            messages=[message],
            think=False,
            options={**options},
        )
        content = response.message.content
        elapsed = time.perf_counter() - t0

        log(
            "✅",
            "FULL RESPONSE",
            f"len={len(content)} chars  time={elapsed:.2f}s",
        )
        print(f"  [full response] {content[:300]}{'...' if len(content) > 300 else ''}")

        return JSONResponse({"response": content})

    except Exception as e:
        elapsed = time.perf_counter() - t0
        log("❌", "FULL RESPONSE ERROR", f"{str(e)}  [{elapsed:.2f}s]")
        raise HTTPException(status_code=500, detail=str(e))