# Gobed Example

This example demonstrates how to use the [gobed](https://github.com/lee101/gobed) embedding model library.

## Setup

1. First, ensure the gobed model files are available:

```bash
# Create symlink to model files from gobed
ln -s ../gobed/model model

# Or copy the model files
cp -r ../gobed/model .
```

2. Install dependencies:

```bash
go mod tidy
```

## Running the Example

```bash
go run main.go
```

## What This Example Shows

- **Loading the Model**: How to load the pre-trained embedding model
- **Text Encoding**: Converting text to high-dimensional embeddings
- **Similarity Calculation**: Computing semantic similarity between texts
- **Finding Similar Texts**: Searching for the most similar texts to a query
- **Performance Benchmarking**: Measuring encoding speed and throughput

## Expected Output

The example will:
1. Load the static-retrieval-mrl-en-v1 model (119MB)
2. Show available pre-tokenized texts
3. Encode sample texts and display their embeddings
4. Calculate similarity scores between text pairs
5. Find the most similar texts to a query
6. Run a performance benchmark showing ~150,000+ encodings/second

## Model Requirements

The gobed library requires:
- The safetensors model file (~119MB)
- Pre-computed tokenization data
- Go 1.21 or later

## Performance

On a typical CPU, you can expect:
- **Model loading**: ~250-500ms
- **Single encoding**: ~6-15μs
- **Throughput**: 70,000-150,000 encodings/second
- **Memory usage**: ~120MB

This is approximately 71x faster than Python GPU inference!