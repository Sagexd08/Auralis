FROM rust:1-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake libclang-dev libasound2-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY runtime ./runtime
COPY apps/desktop/src-tauri/Cargo.toml ./apps/desktop/src-tauri/Cargo.toml
RUN mkdir -p apps/desktop/src-tauri/src && echo "fn main() {}" > apps/desktop/src-tauri/src/main.rs \
    && echo "" > apps/desktop/src-tauri/src/lib.rs
RUN cargo build --release --locked -p auralis-runtime --bin auralis-server --bin auralis-router

FROM debian:bookworm-slim AS runtime-base
RUN apt-get update && apt-get install -y --no-install-recommends libasound2 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home auralis
USER auralis
WORKDIR /home/auralis

FROM runtime-base AS server
COPY --from=build /src/target/release/auralis-server /usr/local/bin/auralis-server
EXPOSE 8787
ENTRYPOINT ["auralis-server", "--bind", "0.0.0.0", "--port", "8787"]
CMD ["--model", "/models/ggml-base.en-q5_1.bin"]

FROM runtime-base AS router
COPY --from=build /src/target/release/auralis-router /usr/local/bin/auralis-router
EXPOSE 8780
ENTRYPOINT ["auralis-router", "--bind", "0.0.0.0", "--port", "8780"]
