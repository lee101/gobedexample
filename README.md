# Gobed Example

A simple example showing how to use the [gobed](https://github.com/lee101/gobed) embedding library.

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

## Setup

```bash
# Clone this example
git clone <your-repo> gobedexample
cd gobedexample

# Download model files (one-time setup)
git clone https://github.com/lee101/gobed ../gobed
cd ../gobed && ./setup.sh && cd ../gobedexample
ln -s ../gobed/model model

# Run the example
go run main.go
```

## What it does

1. **Loads the model** - ~250ms startup time
2. **Encodes text** - Converts text to 1024-dimensional vectors
3. **Calculates similarity** - Measures semantic similarity between texts
4. **Finds similar texts** - Searches for most similar texts
5. **Benchmarks performance** - Shows ~150,000+ encodings/second

## Using in your own project

```bash
go get github.com/lee101/gobed
```

Then copy the model files to your project or set up a symlink as shown above.

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