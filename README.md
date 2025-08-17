# Gobed Example

A simple example showing how to use the [gobed](https://github.com/lee101/gobed) embedding library for text embeddings and similarity calculations.

## 🤖 Model

Gobed uses **[sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1)**, a high-performance static embedding model that:

- Generates 1024-dimensional embeddings
- Optimized for retrieval and similarity tasks
- Supports English text
- Runs efficiently on CPU (no GPU required)
- Provides excellent semantic understanding for text similarity

## 🚀 Quick Start

### Installation

```bash
go get github.com/lee101/gobed
```

### Basic Usage

```go
package main

import (
    "fmt"
    "log"
    "github.com/lee101/gobed"
)

func main() {
    // Load the embedding model
    model, err := gobed.LoadModel()
    if err != nil {
        log.Fatal("Failed to load model:", err)
    }
    defer model.Close()
    
    // Calculate similarity between two texts
    text1 := "Machine learning is fascinating"
    text2 := "Deep learning is powerful"
    
    similarity := model.Similarity(text1, text2)
    fmt.Printf("Similarity: %.4f\n", similarity)
}
```

## 📚 Core API

### LoadModel()
Loads the embedding model from disk. The model must be available in a `model/` directory.

```go
model, err := gobed.LoadModel()
if err != nil {
    log.Fatal(err)
}
defer model.Close()  // Always close when done
```

### Embed()
Converts text into a 1024-dimensional embedding vector.

```go
// Single text embedding
embedding := model.Embed("Your text here")
// Returns []float32 with 1024 dimensions

// Batch embedding for multiple texts
texts := []string{"Text one", "Text two", "Text three"}
embeddings := model.EmbedBatch(texts)
// Returns [][]float32
```

### Similarity()
Calculates cosine similarity between two texts (0 to 1, where 1 = identical).

```go
similarity := model.Similarity("Text A", "Text B")
// Returns float32 between 0 and 1

// Interpretation:
// > 0.9  = Very similar
// 0.7-0.9 = Similar  
// 0.4-0.7 = Moderately similar
// < 0.4  = Different
```

## 💡 Common Use Cases

### Semantic Search
Find the most similar documents to a query:

```go
// Your document database
documents := []string{
    "The quick brown fox jumps over the lazy dog",
    "Machine learning transforms data into insights",
    "Natural language processing enables computers to understand text",
    "The weather today is sunny and warm",
}

query := "AI and text understanding"

// Embed query
queryEmbed := model.Embed(query)

// Find most similar document
bestIdx := -1
bestScore := float32(0)

for i, doc := range documents {
    docEmbed := model.Embed(doc)
    score := gobed.CosineSimilarity(queryEmbed, docEmbed)
    if score > bestScore {
        bestScore = score
        bestIdx = i
    }
}

fmt.Printf("Most similar: %s (score: %.3f)\n", documents[bestIdx], bestScore)
```

### Duplicate Detection
Identify near-duplicate content:

```go
threshold := float32(0.85)  // 85% similarity threshold

texts := []string{
    "The car is red",
    "The automobile is red", 
    "The vehicle is crimson",
    "The sky is blue",
}

for i := 0; i < len(texts); i++ {
    for j := i + 1; j < len(texts); j++ {
        sim := model.Similarity(texts[i], texts[j])
        if sim > threshold {
            fmt.Printf("Potential duplicates (%.1f%% similar):\n", sim*100)
            fmt.Printf("  - %s\n  - %s\n", texts[i], texts[j])
        }
    }
}
```

### Clustering Similar Texts
Group texts by semantic similarity:

```go
texts := []string{
    "Python is a programming language",
    "Java is used for software development",
    "Dogs are loyal pets",
    "Cats are independent animals",
    "JavaScript runs in browsers",
}

// Simple clustering by similarity
groups := make(map[int][]string)
assigned := make([]bool, len(texts))
groupId := 0

for i, text := range texts {
    if assigned[i] {
        continue
    }
    
    // Start new group
    groups[groupId] = []string{text}
    assigned[i] = true
    
    // Find similar texts
    for j := i + 1; j < len(texts); j++ {
        if !assigned[j] {
            if model.Similarity(text, texts[j]) > 0.6 {
                groups[groupId] = append(groups[groupId], texts[j])
                assigned[j] = true
            }
        }
    }
    groupId++
}

// Print groups
for id, group := range groups {
    fmt.Printf("Group %d:\n", id)
    for _, text := range group {
        fmt.Printf("  - %s\n", text)
    }
}
```

## 🎯 Interactive Similarity Calculator

We've included an interactive CLI tool to test text similarity:

```bash
# Run the similarity calculator
./run_similarity.sh

# Or directly:
go run similarity.go
```

### How to use:
1. Run the tool
2. Enter your first text when prompted
3. Enter your second text
4. See the similarity score, distance, and interpretation
5. Type 'quit' to exit

### Example session:
```
🚀 Gobed Similarity Calculator
================================

Loading model... ✓

Enter first text (or 'quit' to exit): Machine learning is fascinating.
Enter second text: Deep learning models are powerful.

📊 Calculating...

============================================================
📝 Text 1: "Machine learning is fascinating."
📝 Text 2: "Deep learning models are powerful."
------------------------------------------------------------
✨ Cosine Similarity: 0.3333
   → Moderately similar ✓
📏 Distance: 0.6667 (1 - similarity)
📊 Similarity percentage: 33.3%
============================================================
```

## 🔧 Model Setup

The gobed library requires model files to be available. There are two ways to set this up:

### Option 1: Download model files directly
```bash
# Clone gobed repo and run setup
git clone https://github.com/lee101/gobed
cd gobed && ./setup.sh
# This downloads the model files to gobed/model/
```

### Option 2: Link to existing model files
```bash
# If you already have the model files elsewhere
ln -s /path/to/model ./model
```

The model directory should contain:
- `model.onnx` - The ONNX model file
- `tokenizer.json` - The tokenizer configuration

## ⚡ Performance

- **Load time**: ~250ms model initialization
- **Encoding speed**: ~150,000+ texts/second
- **Memory usage**: ~500MB for model
- **Vector dimensions**: 1024 (float32)

## Example output

```
🚀 Gobed Example
================
Loading model... ✓ (0.25s)

📝 Encoding texts:
   ✓ Machine learning is fascinating. → [2.74, 13.40, ...]
   ✓ Deep learning models are powerful. → [0.49, 7.40, ...]
   ...

🎯 Similarity scores:
   'Machine learning is fascinating.'
   'Deep learning models are powerful.'
   → Similarity: 0.3333

⚡ Performance:
   • Latency: 6µs per encoding
   • Throughput: 166667 encodings/sec

✅ Done!
```