# OnticalModel Test Infrastructure

This directory contains Docker services for testing the OnticalModel with LLMs.

## Services

### llm-server
Runs Ollama with the llama3.2:1b model for LLM testing.
- **Port**: 11434 (Ollama API)
- **Container**: `llm-server`

### ray-server
Runs Ray Serve for deploying and testing OnticalModelServer.
- **Ports**:
  - 8265 (Ray dashboard)
  - 8000 (Ray Serve HTTP)
  - 10001 (Ray client)
- **Container**: `ray-server`
- **Mounts**: Project root at `/workspace`

## Usage

### Start the services:
```bash
cd test/fixtures/ontical-test-llm
docker-compose up -d
```

### Check service status:
```bash
docker-compose ps
```

### View logs:
```bash
docker-compose logs -f ray-server
docker-compose logs -f llm-server
```

### Run tests:
The tests will automatically connect to the running Ray server:
```bash
pytest test/test_model.py -v
```

### Stop the services:
```bash
docker-compose down
```

### Clean up (including volumes):
```bash
docker-compose down -v
```

## Access Points

- **Ray Dashboard**: http://localhost:8265
- **Ray Serve**: http://localhost:8000
- **Ollama API**: http://localhost:11434

## Environment Variables

- `RAY_ADDRESS`: Ray server address (default: `ray://localhost:10001`)
- `LLM_HOST`: LLM server hostname (set to `llm-server` in Docker, `localhost` outside)

## Architecture

The test setup uses:
1. **Ollama** for the LLM backend
2. **Ray Serve** for deploying OnticalModelServer
3. **pytest** fixtures that connect to the external Ray server and deploy test applications

This architecture mirrors a production deployment while allowing for fast, repeatable testing.

## Troubleshooting

### Tests hang or timeout with "Failed to connect to Ray server"

If tests hang or fail with connection errors, check if you have a local Ray instance running
that conflicts with the Docker Ray server:

```bash
# Check for local Ray processes
ps aux | grep ray

# If found, stop the local Ray instance
ray stop
```

Both the local and Docker Ray servers use port 10001 for the client server. Stop any local
Ray instances before running tests.
