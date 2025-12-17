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

# Combine and view coverage
uv run coverage combine .coverage-data/
uv run coverage report
uv run coverage html  # Generate HTML report in htmlcov/
```

Stop the test environment:

```bash
docker compose -f test/fixtures/ontical-test-llm/docker-compose.yml down
```

### Continuous Integration

[![Test](https://github.com/wpm/OnticalModel/actions/workflows/test.yml/badge.svg)](https://github.com/wpm/OnticalModel/actions/workflows/test.yml)

GitHub Actions automatically runs the test suite on every push and pull request. The CI environment uses the exact same Docker Compose setup as local development, ensuring consistency between local and CI testing.

Coverage reports are automatically uploaded to [Codecov](https://codecov.io) for tracking test coverage over time. Coverage is collected from integration tests by instrumenting the Ray Serve deployment running in Docker containers.
