use axum::{
    extract::{Json, State},
    http::{header, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

mod embedding;
mod index;

use embedding::EmbeddingService;
use index::VectorIndex;

#[derive(Clone)]
struct AppState {
    embedding_service: Arc<EmbeddingService>,
    vector_index: Arc<VectorIndex>,
}

#[derive(Deserialize)]
struct EmbedRequest {
    text: String,
}

#[derive(Serialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
    dimensions: usize,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Serialize, Clone)]
struct SearchResult {
    id: String,
    score: f32,
    text: String,
}

#[derive(Deserialize)]
struct IndexRequest {
    id: String,
    text: String,
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct IndexResponse {
    success: bool,
    id: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    dimensions: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug)]
enum AppError {
    EmbeddingError(String),
    NotFound(String),
    BadRequest(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::EmbeddingError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
        };

        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::EmbeddingError(err.to_string())
    }
}

// Handlers
async fn health(State(_state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: "all-MiniLM-L6-v2".to_string(),
        dimensions: 384,
    })
}

async fn embed(
    State(state): State<AppState>,
    Json(payload): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, AppError> {
    let embedding = state.embedding_service.embed(&payload.text).await?;

    Ok(Json(EmbedResponse {
        dimensions: embedding.len(),
        embedding,
    }))
}

async fn index_document(
    State(state): State<AppState>,
    Json(payload): Json<IndexRequest>,
) -> Result<Json<IndexResponse>, AppError> {
    let embedding = state.embedding_service.embed(&payload.text).await?;

    state
        .vector_index
        .add(&payload.id, embedding, payload.text, payload.metadata)
        .await?;

    Ok(Json(IndexResponse {
        success: true,
        id: payload.id,
    }))
}

async fn search(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    let query_embedding = state.embedding_service.embed(&payload.query).await?;

    let limit = payload.limit.unwrap_or(10);
    let results = state
        .vector_index
        .search(&query_embedding, limit)
        .await?;

    Ok(Json(SearchResponse { results }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("systematics_embeddings=info,tower_http=debug")
        .init();

    info!("Starting Systematics Embedding Server");

    // Initialize embedding service
    info!("Loading embedding model...");
    let embedding_service = Arc::new(EmbeddingService::new().await?);
    info!("Embedding model loaded successfully");

    // Initialize vector index
    let vector_index = Arc::new(VectorIndex::new());

    let state = AppState {
        embedding_service,
        vector_index,
    };

    // Configure CORS for Obsidian
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([header::CONTENT_TYPE]);

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/embed", post(embed))
        .route("/index", post(index_document))
        .route("/search", post(search))
        .layer(cors)
        .with_state(state);

    // Start server
    let addr = "127.0.0.1:8765";
    info!("Server listening on {}", addr);
    println!("🚀 Systematics Embedding Server ready at http://{}", addr);
    println!("   - Health check: GET  http://{}/health", addr);
    println!("   - Embed text:   POST http://{}/embed", addr);
    println!("   - Index doc:    POST http://{}/index", addr);
    println!("   - Search:       POST http://{}/search", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
