# Integration Tests

**Integration tests run ONLY in CI, never locally.**

These tests verify the full system working together with actual Docker containers
running Redis, Ray Serve, and the Ollama LLM.

## For Local Development

**Use unit tests instead:**
```bash
uv run pytest test/unit/ -v
```

Unit tests use mocks and run instantly without any infrastructure.

## CI Execution

GitHub Actions automatically:
1. Starts Docker containers (Redis, Ollama LLM, Ray Serve)
2. Waits for services to be healthy
3. Runs integration tests
4. Tears down containers

See `.github/workflows/test.yml` for the full CI configuration.

## Infrastructure

Integration tests require:
- **Redis** - Persistent checkpoint storage
- **Ollama LLM** - Language model (llama3.2:1b)
- **Ray Serve** - Distributed serving infrastructure
- **OnticalModelServer** - Deployed application

All services are defined in `test/fixtures/ontical-test-llm/docker-compose.yml`

## Test Structure

- `test_langgraph_agent.py` - Tests LangGraphAgent with actual LLM inference
- `test_ray_serve.py` - Tests Ray Serve deployment via HTTP endpoints
- `conftest.py` - Fixtures assuming CI environment
