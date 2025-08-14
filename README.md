# Gobed Example

This example demonstrates how to use the [gobed](https://github.com/lee101/gobed) embedding model library.

## Prerequisites

- Go 1.21 or later
- The gobed model files (see setup below)

## Setup

### Option 1: Clone and Run

```bash
# Clone this example repository
git clone <your-repo-url> gobedexample
cd gobedexample

# The go.mod already includes the gobed dependency from GitHub
# It will be automatically downloaded when you run the example

# Create symlink to model files (if you have gobed cloned locally)
ln -s /path/to/gobed/model model

# Or download the model files using the gobed setup script
git clone https://github.com/lee101/gobed.git ../gobed
cd ../gobed && ./setup.sh && cd ../gobedexample
ln -s ../gobed/model model
```

### Option 2: Use in Your Own Project

```bash
# In your Go project, get the gobed library
go get github.com/lee101/gobed@latest

# You'll need to have the model files available
# Download them using the gobed setup script or copy from an existing installation
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