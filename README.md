# Gobed Example

Simple example showing how to use [gobed](https://github.com/lee101/gobed) for text embeddings and similarity search.

## Quick Start

```bash
# Install dependencies
go mod tidy

# Run basic example
go run main.go

# Interactive similarity calculator
go run similarity.go
```

## Basic Usage

```go
package main

import (
    "fmt"
    "log"
    "github.com/lee101/gobed"
)

func main() {
    // Load model
    model, err := gobed.LoadModel()
    if err != nil {
        log.Fatal(err)
    }
    defer model.Close()
    
    // Calculate similarity
    similarity := model.Similarity(
        "Machine learning is fascinating",
        "Deep learning is powerful",
    )
    
    fmt.Printf("Similarity: %.3f\n", similarity)
    // Output: Similarity: 0.333
}
```

## What This Example Shows

- **Text similarity**: Compare how similar two pieces of text are (0-1 scale)
- **Batch embedding**: Process multiple texts efficiently  
- **Distance metrics**: Cosine, Euclidean, and Manhattan distances
- **Interactive CLI**: Test similarity calculations in real-time

## Model Details

Uses **sentence-transformers/static-retrieval-mrl-en-v1**:
- 1024-dimensional embeddings
- Optimized for retrieval tasks
- CPU-only (no GPU required)
- ~150K texts/second encoding speed

## Common Use Cases

**Find similar documents:**
```go
query := "machine learning"
bestMatch := ""
bestScore := float32(0)

for _, doc := range documents {
    score := model.Similarity(query, doc)
    if score > bestScore {
        bestScore = score
        bestMatch = doc
    }
}
```

**Detect duplicates:**
```go
threshold := float32(0.85)
for i := 0; i < len(texts); i++ {
    for j := i + 1; j < len(texts); j++ {
        if model.Similarity(texts[i], texts[j]) > threshold {
            fmt.Printf("Potential duplicate: %s vs %s\n", texts[i], texts[j])
        }
    }
}
```

**Group similar content:**
```go
// Embed all texts once
embeddings := model.EmbedBatch(texts)

// Find clusters using similarity threshold
for i, emb1 := range embeddings {
    for j, emb2 := range embeddings[i+1:] {
        sim := gobed.CosineSimilarity(emb1, emb2)
        if sim > 0.7 {
            fmt.Printf("Similar: %s <-> %s (%.3f)\n", 
                texts[i], texts[i+j+1], sim)
        }
    }
}
```

## Interactive Tool

Test similarity between any two texts:

```bash
./run_similarity.sh
# or
go run similarity.go
```

Example session:
```
🚀 Gobed Similarity Calculator
Enter first text: The weather is nice today
Enter second text: It's a beautiful sunny day

✨ Similarity: 0.745 (highly similar)
📏 Distance: 0.255
```

## Setup

The model files need to be available:

```bash
# Option 1: Download directly
git clone https://github.com/lee101/gobed
cd gobed && ./setup.sh

# Option 2: Link existing model
ln -s /path/to/model ./model
```

## Performance

- **Load time**: ~250ms
- **Encoding**: 150K+ texts/second  
- **Memory**: ~500MB
- **Embedding size**: 1024 dimensions

## Example Output

```
🚀 Gobed Example
Loading model... ✓ (0.25s)

📝 Similarity between related texts:
   'Machine learning is fascinating' vs 'AI will change the world'
   → 0.431 (moderately similar)

📝 Similarity between unrelated texts:  
   'Programming in Python' vs 'Pizza tastes good'
   → 0.089 (very different)

⚡ Performance: 166K encodings/sec
```

## Files

- `main.go` - Basic similarity examples with distance metrics
- `similarity.go` - Interactive CLI calculator
- `run_similarity.sh` - Convenience script