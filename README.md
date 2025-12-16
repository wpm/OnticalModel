# Ontical Model

Language model services for Ontical entities, hosted using Ray Serve.

## Installation

```bash
pip install -e .
```

## Running the Service

Start the model server using the provided configuration:

```bash
serve run serve_config.yaml
```

## Development

Install test dependencies:

```bash
pip install -r test-requirements.txt
```

Start the test environment:

```bash
cd test/fixtures/ontical-test-llm
docker-compose up -d
```

Run tests:

```bash
pytest
```
