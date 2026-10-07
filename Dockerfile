# Container for the web prototype.
#
# ngspice is the only system dependency that matters, and it is why this needs
# a container rather than a static host: every page that shows a truth table
# runs a real simulation.

FROM python:3.12-slim

# ngspice for simulation, git for the PDK fetch.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ngspice git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The PDK fetch comes before the source copy so that editing bias/ does not
# re-download it. Only the two primitives bias instantiates, not the 770MB
# library.
COPY scripts ./scripts
COPY pdks ./pdks
RUN chmod +x scripts/fetch_pdk.sh && scripts/fetch_pdk.sh sky130 \
 && rm -rf pdks/sky130/.git

COPY pyproject.toml README.md ./
COPY bias ./bias
RUN pip install --no-cache-dir -e '.[web]'

COPY web ./web

# Fail the build rather than ship an image whose simulator is broken.
RUN python -c "\
from bias import pdk, verify;\
assert 'sky130' in pdk.available(), pdk.available();\
r = verify.verify('half_adder', pdk.get('sky130'));\
assert r.passed, r.reason;\
print('sky130 verified in image:', r.transistors, 'transistors')"

ENV PORT=8010
EXPOSE 8010
CMD ["sh", "-c", "uvicorn web.server:app --host 0.0.0.0 --port ${PORT}"]
