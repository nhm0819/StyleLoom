# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

# ffmpeg  -- media.py shells out to ffmpeg and ffprobe for every keyframe
#            animation, caption burn, concat and duration probe. Not optional.
# fonts-noto-cjk -- drawtext needs a font file that actually contains Hangul or
#            Korean captions render as empty boxes with no error.
#            media.resolve_font() looks for
#            /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc first, which is
#            exactly what this package installs.
#
# Nothing else is needed. opencv-python-headless links only libc, libstdc++,
# libgcc_s, libz and libm, all already in slim -- verified with ldd on the
# installed wheel. That is the reason the dependency is pinned to the headless
# build: the normal opencv-python wheel pulls in libGL, libX11 and four Qt5
# libraries that would all have to be installed here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# WORKDIR is load-bearing, not cosmetic. Two relative paths resolve against the
# process cwd:
#   Settings.data_dir  = Path("data")   -> /app/data
#   SettingsConfigDict(env_file=".env") -> /app/.env
# Change this line and runs are written somewhere else and .env stops being read,
# both silently.
WORKDIR /app

COPY . /app

# Editable install, deliberately, even though nothing gets edited inside an image.
# config.py computes REPO_ROOT as Path(__file__).resolve().parents[2], which
# resolve_config() uses to fall back to the bundled configs/*.yaml. Under a normal
# site-packages install that expression evaluates to /usr/local/lib/python3.12, so
# the fallback points at a directory that does not exist:
#   ConfigError: config file not found: configs/archetypes.yaml
#     (also tried /usr/local/lib/python3.12/configs/archetypes.yaml)
# An editable install keeps REPO_ROOT at /app, where configs/ actually is.
#
# No extras: the kling provider signs its own JWT and posts with httpx, so
# KLING_ACCESS_KEY and KLING_SECRET_KEY in .env are enough to reach real video
# generation. Without them the provider cannot construct and the image is
# offline-only.
RUN python -m pip install -e agent-core -e cli

# Runs write into data/. Root-owned output on a bind mount is a nuisance to clean
# up from the host, so drop to a normal user. UID 1000 matches the usual first
# host user; override with `--user "$(id -u):$(id -g)"` when it does not.
RUN useradd --create-home --uid 1000 styleloom \
    && mkdir -p data/runs data/styles data/uploads \
    && chown -R styleloom:styleloom /app/data
USER styleloom

# `docker run <image> run my_style --text "..."` reads as the CLI it wraps.
# CMD gives a bare `docker run <image>` something useful to print.
ENTRYPOINT ["styleloom"]
CMD ["--help"]
