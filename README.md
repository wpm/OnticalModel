# Ontical Model

Models used by Ontical entities.

The models are hosted by [Ray Serve](https://docs.ray.io/en/latest/serve/index.html).
Model services are launched using Ray Serve command line tools.

## Architecture

This project uses the Ray Serve **builder pattern** for deployment configuration. The builder pattern allows you to configure deployments using external configuration files or CLI arguments without modifying code.

### Key Components

- **`OnticalModelServer`**: Ray Serve deployment class that hosts a ChatAgent with LLM capabilities
- **`OnticalModelServerArgs`**: Pydantic model for type-safe configuration
- **`app_builder`**: Builder function that creates deployments from configuration
- **`ChatAgent`**: LangGraph-based agent with memory and structured output

## Deployment

### Using YAML Configuration (Recommended)

The recommended approach is to use a YAML configuration file:

```bash
serve run serve_config.yaml
```

This uses the configuration in `serve_config.yaml`:

```yaml
applications:
  - name: ontical-model-server
    import_path: ontical_model.server:app_builder
    route_prefix: /
    args:
      model_name: "llama3.2:1b"
      base_url: "http://llm-server:11434/v1"
      api_key: "ollama"
      schema_class: "ontical_model.schemas.Colors"
      initial_prompt: "Answer questions accurately and succinctly."
      temperature: 0.7
      max_tokens: 150
```

### Using CLI Arguments

You can also pass configuration directly via CLI:

```bash
serve run ontical_model.server:app_builder \
  model_name="llama3.2:1b" \
  base_url="http://localhost:11434/v1" \
  schema_class="ontical_model.schemas.Colors" \
  initial_prompt="Answer questions accurately and succinctly." \
  temperature=0.7 \
  max_tokens=150
```

### Configuration Parameters

- **`model_name`**: The name of the LLM model (e.g., "llama3.2:1b")
- **`base_url`**: OpenAI-compatible API endpoint (e.g., Ollama server)
- **`api_key`**: API key (use "ollama" for local Ollama servers)
- **`schema_class`**: Fully qualified Pydantic schema class name for structured output
- **`initial_prompt`**: System prompt for the agent
- **`temperature`**: Model temperature (0.0-1.0)
- **`max_tokens`**: Maximum response length

## Development

### Installation

```bash
pip install -e .
```

### Running Tests

```bash
# Install test dependencies
pip install -r test-requirements.txt

# Start the test environment (Ollama + Ray + Prometheus + Grafana)
cd test/fixtures/ontical-test-llm
docker-compose up -d

# Run tests
pytest
```

### Testing Infrastructure

The test environment includes:
- **Ollama LLM Server** (port 11434): OpenAI-compatible API
- **Ray Serve** (port 8000): Deployment platform
- **Ray Dashboard** (port 8265): Monitoring and observability
- **Prometheus** (port 9090): Metrics collection
- **Grafana** (port 3001): Dashboards and visualization

## Benefits of the Builder Pattern

1. **Configuration without code changes**: Modify deployment parameters via config files
2. **Type safety**: Pydantic validation ensures correct configuration
3. **Reusability**: Same code works with different configurations
4. **External configuration**: Enable GitOps and environment-specific deployments
5. **CLI-friendly**: Easy to override specific parameters from command line
