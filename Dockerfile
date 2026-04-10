FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    libopenblas-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN cmake -S . -B build/docker-ui -G Ninja -DQASR_ENABLE_TESTS=OFF \
    && cmake --build build/docker-ui -j"$(nproc)"

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    alsa-utils \
    ca-certificates \
    curl \
    ffmpeg \
    libopenblas0-pthread \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /workspace/build/docker-ui/qasr_server /app/qasr_server
COPY --from=build /workspace/build/docker-ui/qasr_cli /app/qasr_cli
COPY ui /app/ui

EXPOSE 8080

ENTRYPOINT ["/app/qasr_server"]
CMD ["--model-dir", "/models/qwen3-asr", "--ui-dir", "/app/ui", "--host", "0.0.0.0", "--port", "8080"]
