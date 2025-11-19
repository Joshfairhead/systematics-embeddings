use anyhow::{Context, Result};
use ndarray::{Array, ArrayView};
use ort::{
    session::{Session, SessionOutputs},
    value::Value,
    GraphOptimizationLevel,
};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tracing::info;

pub struct EmbeddingService {
    session: Session,
    tokenizer: Tokenizer,
}

impl EmbeddingService {
    pub async fn new() -> Result<Self> {
        // Download and load model
        let model_path = Self::download_model().await?;

        info!("Loading ONNX model from {:?}", model_path);
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&model_path)?;

        info!("Loading tokenizer");
        let tokenizer_path = Self::download_tokenizer().await?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .context("Failed to load tokenizer")?;

        Ok(Self { session, tokenizer })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, false)
            .context("Failed to tokenize")?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Convert to arrays
        let input_ids_array = Array::from_shape_vec((1, input_ids.len()), input_ids.to_vec())?;
        let attention_mask_array =
            Array::from_shape_vec((1, attention_mask.len()), attention_mask.to_vec())?;

        // Run inference
        let outputs: SessionOutputs = self.session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids_array)?,
            "attention_mask" => Value::from_array(attention_mask_array)?,
        ]?)?;

        // Extract embeddings (last_hidden_state)
        let embeddings = outputs[0]
            .try_extract_tensor::<f32>()?
            .view()
            .to_owned();

        // Mean pooling
        let pooled = self.mean_pooling(&embeddings, attention_mask);

        // Normalize
        let normalized = self.normalize(&pooled);

        Ok(normalized)
    }

    fn mean_pooling(&self, embeddings: &ArrayView<f32, ndarray::IxDyn>, attention_mask: &[u32]) -> Vec<f32> {
        let shape = embeddings.shape();
        let seq_len = shape[1];
        let hidden_size = shape[2];

        let mut pooled = vec![0.0f32; hidden_size];
        let mut mask_sum = 0.0f32;

        for i in 0..seq_len {
            if attention_mask[i] == 1 {
                for j in 0..hidden_size {
                    pooled[j] += embeddings[[0, i, j]];
                }
                mask_sum += 1.0;
            }
        }

        // Average
        for val in &mut pooled {
            *val /= mask_sum;
        }

        pooled
    }

    fn normalize(&self, vec: &[f32]) -> Vec<f32> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        vec.iter().map(|x| x / norm).collect()
    }

    async fn download_model() -> Result<PathBuf> {
        // For now, assume model is in models/ directory
        // In production, download from HuggingFace
        let model_dir = PathBuf::from("models");
        std::fs::create_dir_all(&model_dir)?;

        let model_path = model_dir.join("model.onnx");

        if !model_path.exists() {
            info!("Downloading model from HuggingFace...");
            // TODO: Download model
            // For now, return error with instructions
            anyhow::bail!(
                "Model not found. Please download model manually:\n\
                 1. Download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2\n\
                 2. Convert to ONNX format\n\
                 3. Place in: {:?}",
                model_path
            );
        }

        Ok(model_path)
    }

    async fn download_tokenizer() -> Result<PathBuf> {
        let model_dir = PathBuf::from("models");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !tokenizer_path.exists() {
            anyhow::bail!(
                "Tokenizer not found. Please download:\n\
                 1. Download tokenizer.json from HuggingFace\n\
                 2. Place in: {:?}",
                tokenizer_path
            );
        }

        Ok(tokenizer_path)
    }
}
