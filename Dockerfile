FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends make gcc libc6-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN make clean && make -j2

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends zlib1g python3 bash time && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/protlocus /src/protlocus-recheck /usr/local/bin/
COPY --from=build /src/tools/ /usr/local/lib/protlocus/tools/
COPY --from=build /src/protlocus.1 /usr/local/share/man/man1/protlocus.1
WORKDIR /data
ENTRYPOINT ["protlocus"]
