# Systematics Embedding Server

A standalone Rust server that provides semantic embeddings for the Obsidian Systematics plugin.

## Features

- ✅ Local embedding generation (no API costs)
- ✅ Fast: ~500 embeddings/second
- ✅ Low memory: ~300-500MB
- ✅ Single binary distribution
- ✅ REST API for easy integration
- ✅ Works with any data source (vaults, websites, etc.)

## Installation

### Prerequisites

- Rust 1.70+ (install from https://rustup.rs)
- ~100MB disk space for model

### Setup

1. **Download the model**:
   ```bash
   ./download-model.sh
   ```

   This downloads the `all-MiniLM-L6-v2` model from HuggingFace and converts it to ONNX format.

2. **Build the server**:
   ```bash
   cargo build --release
   ```

3. **Run the server**:
   ```bash
   ./target/release/systematics-embeddings
   ```

The server will start on `http://localhost:8765`.

## API Endpoints

### Health Check
```bash
GET /health

Response:
{
  "status": "ok",
  "model": "all-MiniLM-L6-v2",
  "dimensions": 384
}
```

### Generate Embedding
```bash
POST /embed
Content-Type: application/json

{
  "text": "Your text here"
}

Response:
{
  "embedding": [0.123, -0.456, ...],
  "dimensions": 384
}
```

### Index Document
```bash
POST /index
Content-Type: application/json

{
  "id": "note-path",
  "text": "Note content",
  "metadata": { "title": "My Note" }
}

Response:
{
  "success": true,
  "id": "note-path"
}
```

### Search
```bash
POST /search
Content-Type: application/json

{
  "query": "semantic search query",
  "limit": 10
}

Response:
{
  "results": [
    {
      "id": "note-path",
      "score": 0.95,
      "text": "Note content snippet"
    }
  ]
}
```

## Usage with Obsidian Plugin

1. Start this server: `./target/release/systematics-embeddings`
2. Open Obsidian
3. Enable the Systematics plugin
4. The plugin will automatically connect to `localhost:8765`
5. Click "Index Vault" to start indexing

## Development

```bash
# Run in development mode with hot reload
cargo watch -x run

# Run tests
cargo test

# Build optimized release binary
cargo build --release
```

## Architecture

```
┌─────────────────────────────────┐
│   Embedding Server (Rust)       │
│                                  │
│   ┌──────────────────────────┐  │
│   │  ONNX Runtime            │  │
│   │  (all-MiniLM-L6-v2)      │  │
│   └──────────────────────────┘  │
│                                  │
│   ┌──────────────────────────┐  │
│   │  Vector Index (in-memory)│  │
│   │  Cosine Similarity Search│  │
│   └──────────────────────────┘  │
│                                  │
│   ┌──────────────────────────┐  │
│   │  REST API (Axum)         │  │
│   │  localhost:8765          │  │
│   └──────────────────────────┘  │
└─────────────────────────────────┘
         ↕ HTTP
┌─────────────────────────────────┐
│   Obsidian Plugin (TypeScript)  │
└─────────────────────────────────┘
```

## Performance

- **Embedding generation**: ~500 embeddings/second
- **Search**: <10ms for 10k documents
- **Memory**: ~300-500MB
- **Startup**: <1 second

## Troubleshooting

### Model not found
Run `./download-model.sh` to download the embedding model.

### Port 8765 already in use
Change the port in `src/main.rs` (line ~195) and rebuild.

### ONNX Runtime errors
Make sure you have the ONNX Runtime installed. On macOS/Linux it's included in the `ort` crate.

## License

MIT
