use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::SearchResult;

#[derive(Clone)]
struct IndexedDocument {
    id: String,
    embedding: Vec<f32>,
    text: String,
    metadata: Option<Value>,
}

pub struct VectorIndex {
    documents: RwLock<HashMap<String, IndexedDocument>>,
}

impl VectorIndex {
    pub fn new() -> Self {
        Self {
            documents: RwLock::new(HashMap::new()),
        }
    }

    pub async fn add(
        &self,
        id: &str,
        embedding: Vec<f32>,
        text: String,
        metadata: Option<Value>,
    ) -> Result<()> {
        let doc = IndexedDocument {
            id: id.to_string(),
            embedding,
            text,
            metadata,
        };

        let mut docs = self.documents.write().unwrap();
        docs.insert(id.to_string(), doc);

        Ok(())
    }

    pub async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let docs = self.documents.read().unwrap();

        let mut results: Vec<SearchResult> = docs
            .values()
            .map(|doc| {
                let score = cosine_similarity(query_embedding, &doc.embedding);
                SearchResult {
                    id: doc.id.clone(),
                    score,
                    text: doc.text.clone(),
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Return top k
        results.truncate(limit);

        Ok(results)
    }

    pub async fn get(&self, id: &str) -> Result<Option<IndexedDocument>> {
        let docs = self.documents.read().unwrap();
        Ok(docs.get(id).cloned())
    }

    pub async fn delete(&self, id: &str) -> Result<bool> {
        let mut docs = self.documents.write().unwrap();
        Ok(docs.remove(id).is_some())
    }

    pub async fn clear(&self) -> Result<()> {
        let mut docs = self.documents.write().unwrap();
        docs.clear();
        Ok(())
    }

    pub async fn count(&self) -> usize {
        let docs = self.documents.read().unwrap();
        docs.len()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
    }
}
