#!/bin/bash
# Startup script for the llm-server service in docker-compose.yml
#
# This script starts the Ollama server, waits for it to be ready, pulls the
# llama3.2:1b model, and keeps the server running. It's used as the CMD in
# Dockerfile.llm-server.

set -e

# Start Ollama in the background
/bin/ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in {1..60}; do
  if curl -f -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama is ready!"
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "Timeout waiting for Ollama to start"
    exit 1
  fi
  echo "Attempt $i/60..."
  sleep 2
done

# Pull the llama3.2:1b
/bin/ollama pull llama3.2:1b

# Keep the server running
echo "LLM service is ready!"
wait $OLLAMA_PID
