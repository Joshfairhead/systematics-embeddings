#!/bin/bash

# Systematics Embedding Server Start Script
# This script activates the virtual environment and starts the server

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Systematics Embedding Server...${NC}\n"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Please run this script from the systematics-embeddings directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}📦 Activating Python virtual environment...${NC}"
    source venv/bin/activate
fi

# Build the server if it doesn't exist
if [ ! -f "target/release/systematics-embeddings" ]; then
    echo -e "${YELLOW}🔨 Building Rust server (first time only, takes a few minutes)...${NC}"
    cargo build --release
    echo -e "${GREEN}✅ Build complete!${NC}\n"
fi

# Check if model exists
if [ ! -d "model" ] && [ ! -f "model.onnx" ]; then
    echo -e "${YELLOW}⚠️  Warning: Model not found. Run 'python download-model.py' first.${NC}"
    echo -e "${YELLOW}   Attempting to download now...${NC}\n"
    python download-model.py
fi

# Start the server
echo -e "${GREEN}🌐 Starting embedding server on http://localhost:8765${NC}\n"
./target/release/systematics-embeddings



