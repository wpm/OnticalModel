[![Test](https://github.com/wpm/OnticalModel/actions/workflows/test.yml/badge.svg)](https://github.com/wpm/OnticalModel/actions/workflows/test.yml)

# Ontical Model

Language model services for Ontical entities, hosted using Ray Serve.

## Installation

```bash
uv pip install -e .
```

**Note:** This package is designed to be deployed as part of a larger Ray Serve application. It does not include a standalone server configuration.

## Development

This package is designed to be deployed as part of a larger application. For local testing and development, use the Docker Compose test environment.

### Setup

Install the package with development dependencies:

```bash
uv sync --dev
```

### Docker Images

The test environment uses pre-built Docker images hosted on GitHub Container Registry for faster startup. Images are automatically rebuilt when dependencies or Dockerfiles change.

**To use pre-built images** (default):
```bash
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml pull
```

**To build images locally** (for development):
```bash
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml build
```

**To manually trigger image rebuild in CI**:
```bash
gh workflow run build-images.yml
```

Images are automatically rebuilt when any of these files change:
- `pyproject.toml` or `uv.lock` (Python dependencies)
- `test/fixtures/ontical-test-llm/Dockerfile.*`
- `test/fixtures/ontical-test-llm/start-ollama.sh`

### Running Tests Locally

Start the test environment with Docker Compose:

```bash
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml up -d
```

The Docker environment includes:
- **llm-server**: Ollama server with llama3.2:1b model
- **ray-server**: Ray cluster head node
- **ontical-model-server**: Ray Serve deployment of OnticalModelServer
- **prometheus**: Metrics collection
- **grafana**: Metrics visualization (available at http://localhost:3001)

Wait for all services to be healthy (check with `docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml ps`), then run tests:

```bash
uv run pytest
```

To collect coverage from the integration tests:

```bash
# Create coverage data directory
mkdir -p .coverage-data

# Run tests
uv run pytest

# Stop the server to flush coverage data
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml stop ontical-model-server

# On macOS, copy coverage files from container (Docker volume sync issue)
docker cp ray-server:/workspace/.coverage-data/. .coverage-data/

# Combine and view coverage
uv run coverage combine .coverage-data/
uv run coverage report
uv run coverage html  # Generate HTML report in htmlcov/
```

**Coverage Limitations**: Due to how Ray Serve initializes actor replicas, coverage instrumentation captures module-level code (imports, class/function definitions) but not the runtime execution inside Ray actor methods. This means coverage reports show which modules are loaded and which classes are defined, but not the actual execution of `__init__` and `__call__` methods in the Ray Serve deployment. This is a known limitation when testing Ray applications.

Stop the test environment:

```bash
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml down
```

### Code Coverage

Coverage reports are automatically uploaded to [Codecov](https://codecov.io) for tracking test coverage over time. Coverage is collected from integration tests by instrumenting the Ray Serve deployment running in Docker containers.
