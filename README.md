[![Test](https://github.com/wpm/OnticalModel/actions/workflows/test.yml/badge.svg)](https://github.com/wpm/OnticalModel/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/wpm/OnticalModel/branch/develop/graph/badge.svg)](https://codecov.io/gh/wpm/OnticalModel)

# Ontical Model

Language model services for Ontical entities, hosted using Ray Serve.

## Installation

Install the package:

```bash
uv pip install -e .
```

For development, install with dev dependencies:

```bash
uv sync --dev
```

## Testing

The project uses a two-tier testing strategy: fast unit tests for development and comprehensive integration tests for
CI/CD verification.

**Continuous Integration:** GitHub Actions automatically runs both unit and integration tests on every push and pull
request.
Pre-built Docker images are pulled from GitHub Container Registry to speed up CI runs.
Coverage reports are uploaded to [Codecov](https://codecov.io) for tracking over time.

### Unit Testing

Unit tests run quickly without external dependencies, providing high code coverage through mocking.

```bash
# Run unit tests only
uv run pytest test/unit/

# Run with coverage report
uv run pytest test/unit/ --cov=ontical_model --cov-config=.coveragerc --cov-report=term-missing

# Generate HTML coverage report
uv run coverage html  # Opens in htmlcov/index.html
```

### Integration Testing

Integration tests verify the full Ray Serve deployment running in Docker containers.

#### Docker Images

The integration test environment uses pre-built Docker images hosted on GitHub Container Registry for faster startup.
Images are automatically rebuilt when dependencies or Dockerfiles change.

**Pull pre-built images** (default):
```bash
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml pull
```

**Build images locally** (for development):
```bash
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml build
```

**Manually trigger image rebuild in CI**:
```bash
gh workflow run build-images.yml
```

Images are automatically rebuilt when these files change:
- `pyproject.toml` or `uv.lock` (Python dependencies)
- `test/fixtures/ontical-test-llm/Dockerfile.*`
- `test/fixtures/ontical-test-llm/start-ollama.sh`

#### Running Integration Tests

Start the Docker environment:

```bash
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml up -d
```

The Docker environment includes:
- **llm-server**: Ollama server with llama3.2:1b model
- **ray-server**: Ray cluster head node
- **ontical-model-server**: Ray Serve deployment of OnticalModelServer
- **prometheus**: Metrics collection
- **grafana**: Metrics visualization (available at http://localhost:3001)

Wait for all services to be healthy (check with `docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml ps`), 
then run tests:

```bash
# Run integration tests only (~7 seconds)
uv run pytest test/integration/

# Run all tests (unit + integration, ~10 seconds)
uv run pytest test/
```

Stop the Docker environment when done:

```bash
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml down
```
