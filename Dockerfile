FROM python:3.9

# =========================
# USER SETUP (HF requirement)
# =========================
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# =========================
# PYTHON DEPENDENCIES
# =========================
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# =========================
# INSTALL OLLAMA
# =========================
USER root
RUN apt-get update && apt-get install -y curl bash

RUN curl -fsSL https://ollama.com/install.sh | sh

USER user

# =========================
# COPY APP
# =========================
COPY --chown=user . /app

# =========================
# START SCRIPT (IMPORTANT)
# =========================
CMD bash -c "\
ollama serve & \
sleep 5 && \
ollama pull qwen3.5:0.8b && \
uvicorn app:app --host 0.0.0.0 --port 7860"