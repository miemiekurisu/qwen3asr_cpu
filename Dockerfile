# ---- Build stage ----
FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    libopenblas-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

# Use the linux-openblas preset; oneDNN is auto-downloaded during configure.
RUN cmake --preset linux-openblas \
        -DQASR_ENABLE_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build/linux-openblas -j"$(nproc)"

# ---- Runtime stage ----
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /workspace/build/linux-openblas/qasr_server /app/qasr_server
COPY --from=build /workspace/build/linux-openblas/qasr_cli    /app/qasr_cli
COPY ui /app/ui

EXPOSE 8080

ENTRYPOINT ["/app/qasr_server"]
CMD ["--model-dir", "/models/qwen3-asr", "--ui-dir", "/app/ui", "--host", "0.0.0.0", "--port", "8080"]
