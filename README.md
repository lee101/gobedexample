# Gobed Example

A simple example showing how to use the [gobed](https://github.com/lee101/gobed) embedding library.

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