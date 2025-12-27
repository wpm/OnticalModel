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

Integration tests verify the full system with actual Docker containers running Redis, Ray Serve, and the Ollama LLM.

**Default workflow:** Integration tests run automatically in CI on every push/PR. For local development, use fast unit tests instead.

**When to run locally:** Only when debugging issues that require the full deployed system (e.g., Ray Serve deployment, Redis checkpointing, actual LLM inference).

#### Infrastructure

Integration tests require these services:
- **Redis** - Persistent checkpoint storage (redis:6379)
- **Ollama LLM** - Language model server (llama3.2:1b on localhost:11434)
- **Ray Serve** - Distributed serving infrastructure (localhost:8000, ray://localhost:10001)
- **OnticalModelServer** - Deployed application

All services are defined in `test/fixtures/ontical-test-llm/docker-compose.yml`.

#### Docker Images

The integration test environment uses pre-built Docker images from GitHub Container Registry for faster CI startup.
Images are automatically rebuilt when dependencies or Dockerfiles change.

**Pull pre-built images** (recommended):
```bash
cd test/fixtures/ontical-test-llm
docker-compose pull
```

**Build images locally** (for development/debugging):
```bash
cd test/fixtures/ontical-test-llm
docker-compose build
```

**Manually trigger image rebuild in CI**:
```bash
gh workflow run build-images.yml
```

Images rebuild automatically when these files change:
- `pyproject.toml` or `uv.lock` (Python dependencies)
- `test/fixtures/ontical-test-llm/Dockerfile.*`
- `test/fixtures/ontical-test-llm/start-ollama.sh`

#### Running Integration Tests Locally (Optional)

**Note:** This is not required for normal development. Use unit tests for rapid feedback.

1. **Start Docker containers:**
   ```bash
   cd test/fixtures/ontical-test-llm
   docker-compose up -d
   ```

2. **Wait for services to be healthy** (60-90 seconds):
   ```bash
   # Check status
   docker-compose ps

   # All services should show "healthy" status
   ```

3. **Run integration tests:**
   ```bash
   cd ../../..  # Back to project root
   uv run pytest test/integration/ -v
   ```

4. **Stop containers when done:**
   ```bash
   cd test/fixtures/ontical-test-llm
   docker-compose down
   ```

**Troubleshooting:**
- First startup is slow (~2-3 minutes) as Ollama downloads the llama3.2:1b model
- Subsequent runs are faster (~60 seconds) as the model is cached in the `ollama_data` volume
- Check logs if services don't become healthy: `docker-compose logs <service-name>`
- Redis data persists in `redis_data` volume between runs
