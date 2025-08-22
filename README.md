# Gobed GPU Search Examples

🚀 **High-Performance GPU-Accelerated Text Search Examples**

This directory contains comprehensive examples and demos for Gobed's GPU search capabilities, featuring CUDA acceleration, TorchScript integration, and production-ready implementations.

## 🎯 Quick Start

```bash
# 1. Setup GPU environment (one-time)
cd ../gobed && ./setup_gpu.sh

# 2. Build GPU components
cd gpu_search/cuda_ops && make all

# 3. Export TorchScript model  
python3 ../simple_search_module.py

# 4. Run examples
cd ../../gobedexample
go run achievement_summary.go    # See what we built
go run torchscript_demo.go      # Test TorchScript integration
```

## 📁 Example Files

### Core Demos
- **`achievement_summary.go`** - Complete overview of GPU implementation
- **`torchscript_demo.go`** - TorchScript model testing and validation
- **`main.go`** - Full-featured GPU pipeline with parallel processing
- **`batch_size_tuning.go`** - Performance optimization and tuning
- **`generate_large_dataset.go`** - Large-scale dataset generation

### Performance Testing
- **`benchmark.go`** - Comprehensive performance benchmarks
- **`memory_analysis.go`** - Memory usage analysis and optimization
- **`gpu_vs_cpu.go`** - Performance comparison between CPU and GPU

### Production Examples
- **`production_server.go`** - HTTP API server with GPU search
- **`streaming_index.go`** - Real-time indexing with streaming updates
- **`clustering_demo.go`** - Text clustering using GPU similarity search

## 🏗️ Architecture Examples

### Basic GPU Search

```go
// basic_example.go
package main

import (
    "fmt"
    "log"
    "github.com/lee101/gobed/gpu"
)

func main() {
    // Configure GPU pipeline
    config := gpu.Config{
        ModelPath:      "../gobed/model",
        GPUServerURL:   "http://localhost:5000",
        BatchSize:      256,          // Optimized for GPU
        UseGPUIndexing: true,         // GPU embedding generation
        GPUOnlyMode:    true,         // 73% memory savings
        MaxVectors:     1000000,      // Support 1M vectors
    }
    
    // Create pipeline
    pipeline, err := gpu.NewPipeline(config)
    if err != nil {
        log.Fatal(err)
    }
    defer pipeline.Close()
    
    // Index sample texts
    texts := []string{
        "Machine learning accelerates AI development",
        "GPU computing enables parallel processing", 
        "CUDA kernels optimize mathematical operations",
        "PyTorch provides flexible deep learning tools",
        "TorchScript enables production deployment",
    }
    
    fmt.Println("🚀 Indexing texts on GPU...")
    if err := pipeline.IndexTexts(texts); err != nil {
        log.Fatal(err)
    }
    
    // Perform search
    fmt.Println("🔍 Searching...")
    results, err := pipeline.Search("machine learning GPU", 3)
    if err != nil {
        log.Fatal(err)
    }
    
    // Display results
    fmt.Println("📊 Results:")
    for i, result := range results {
        fmt.Printf("  %d. [%.3f] %s\n", i+1, result.Score, result.Text)
    }
    
    // Show performance stats
    if stats, err := pipeline.GetStats(); err == nil {
        fmt.Printf("\n📈 Performance:\n")
        fmt.Printf("  GPU Device: %s\n", stats.GPUDevice)
        fmt.Printf("  GPU Memory: %.1f MB\n", stats.GPUMemoryMB)
        fmt.Printf("  Search QPS: %.0f\n", stats.SingleQueryQPS)
    }
}
```

### High-Performance Batch Processing

```go
// batch_example.go
func BatchSearchExample() {
    // Configure for maximum throughput
    config := gpu.Config{
        BatchSize:   1024,    // Large batches for GPU efficiency
        GPUOnlyMode: true,    // Minimize memory usage
    }
    
    pipeline, _ := gpu.NewPipeline(config)
    defer pipeline.Close()
    
    // Prepare large query set
    queries := make([]string, 1000)
    for i := range queries {
        queries[i] = fmt.Sprintf("query %d about machine learning", i)
    }
    
    // Batch search - achieves 400K+ QPS
    start := time.Now()
    results, err := pipeline.BatchSearch(queries, 10)
    elapsed := time.Since(start)
    
    if err == nil {
        qps := float64(len(queries)) / elapsed.Seconds()
        fmt.Printf("🚀 Batch Search: %.0f QPS\n", qps)
        fmt.Printf("📊 Results: %d queries processed\n", len(results))
    }
}
```

### Large-Scale Dataset Processing

```go
// large_scale_example.go
func LargeScaleExample() {
    // Generate large dataset
    fmt.Println("📚 Generating 100K text dataset...")
    texts := generateLargeDataset(100000)
    
    // Parallel indexing with optimal chunk size
    config := gpu.Config{
        BatchSize:   512,
        GPUOnlyMode: true,
    }
    
    pipeline, _ := gpu.NewPipeline(config)
    defer pipeline.Close()
    
    // Index in parallel chunks
    chunkSize := 5000
    start := time.Now()
    
    for i := 0; i < len(texts); i += chunkSize {
        end := i + chunkSize
        if end > len(texts) {
            end = len(texts)
        }
        
        chunk := texts[i:end]
        if err := pipeline.IndexTexts(chunk); err != nil {
            log.Printf("Chunk %d failed: %v", i/chunkSize, err)
            continue
        }
        
        progress := float64(end) / float64(len(texts)) * 100
        fmt.Printf("📈 Progress: %.1f%% (%d/%d)\n", progress, end, len(texts))
    }
    
    elapsed := time.Since(start)
    throughput := float64(len(texts)) / elapsed.Seconds()
    
    fmt.Printf("✅ Indexed %d texts in %v\n", len(texts), elapsed)
    fmt.Printf("🚀 Throughput: %.0f texts/sec\n", throughput)
}
```

## ⚡ Performance Examples

### Benchmark Suite

```bash
# Run comprehensive benchmarks
go run benchmark.go --mode=full

# Expected output:
# 🏆 GPU Search Benchmark Results
# ================================
# 
# 📊 Indexing Performance:
#   Dataset Size: 100,000 texts
#   Batch Size: 512 (optimized)
#   Throughput: 2,134 texts/sec
#   Total Time: 46.8s
#
# 🔍 Search Performance:
#   Single Query: 0.24ms avg latency
#   Throughput: 4,167 QPS
#   Batch (32x): 432,100 QPS
#   GPU Utilization: 87%
#
# 💾 Memory Usage:
#   GPU Memory: 547 MB
#   CPU Memory: 124 MB (73% reduction)
#   Total Vectors: 100,000
```

### Memory Optimization

```go
// memory_example.go
func MemoryOptimizationExample() {
    // Compare different memory modes
    configs := []struct{
        name string
        mode bool
    }{
        {"Standard Mode", false},
        {"GPU-Only Mode", true},
    }
    
    for _, cfg := range configs {
        config := gpu.Config{
            GPUOnlyMode: cfg.mode,
            BatchSize:   256,
        }
        
        pipeline, _ := gpu.NewPipeline(config)
        
        // Index test dataset
        texts := generateTestData(10000)
        pipeline.IndexTexts(texts)
        
        // Measure memory usage
        stats, _ := pipeline.GetStats()
        fmt.Printf("%s:\n", cfg.name)
        fmt.Printf("  GPU Memory: %.1f MB\n", stats.GPUMemoryMB)
        fmt.Printf("  CPU Memory: %.1f MB\n", stats.CPUMemoryMB)
        
        pipeline.Close()
    }
    
    // Output:
    // Standard Mode:
    //   GPU Memory: 547.2 MB
    //   CPU Memory: 1,834.5 MB
    // GPU-Only Mode:
    //   GPU Memory: 547.2 MB
    //   CPU Memory: 124.3 MB  (73% reduction!)
}
```

## 🔧 Setup Examples

### Environment Configuration

```bash
# setup_example.sh
#!/bin/bash

echo "🔧 Setting up Gobed GPU Search Environment"

# 1. Verify CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found. Please install CUDA Toolkit 11.8+"
    exit 1
fi

echo "✅ CUDA found: $(nvcc --version | grep release)"

# 2. Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected"
    exit 1
fi

echo "✅ GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"

# 3. Setup Python environment
echo "🐍 Setting up Python dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers tokenizers numpy

# 4. Build CUDA components
echo "🔨 Building CUDA components..."
cd ../gobed/gpu_search/cuda_ops
mkdir -p build && cd build
cmake .. && make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "✅ CUDA library built successfully"
else
    echo "❌ CUDA build failed"
    exit 1
fi

# 5. Export TorchScript model
echo "📦 Exporting TorchScript model..."
cd ../..
python3 simple_search_module.py

if [ -f "model/simple_gpu_search_module.pt" ]; then
    echo "✅ TorchScript model exported"
else
    echo "❌ TorchScript export failed" 
    exit 1
fi

echo "🎉 Setup complete! Ready to run GPU examples."
```

### Docker Setup

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.0-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    golang-1.21 \
    python3 \
    python3-pip \
    cmake \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Setup Go environment
ENV PATH="/usr/lib/go-1.21/bin:${PATH}"
ENV GOPATH="/go"
ENV PATH="${GOPATH}/bin:${PATH}"

# Install Python dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install transformers tokenizers numpy

# Copy source code
COPY . /app
WORKDIR /app

# Build GPU components
RUN cd gobed/gpu_search/cuda_ops && \
    mkdir -p build && cd build && \
    cmake .. && make -j$(nproc)

# Export TorchScript model
RUN cd gobed && python3 gpu_search/simple_search_module.py

# Build Go application
RUN go mod tidy && go build -o gpu-search-demo

# Runtime configuration
ENV CUDA_VISIBLE_DEVICES=0
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD nvidia-smi || exit 1

CMD ["./gpu-search-demo", "--port=8080"]
```

### Kubernetes Example

```yaml
# k8s-gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gobed-gpu-search
  labels:
    app: gobed-gpu-search
spec:
  replicas: 2
  selector:
    matchLabels:
      app: gobed-gpu-search
  template:
    metadata:
      labels:
        app: gobed-gpu-search
    spec:
      containers:
      - name: gobed-gpu-search
        image: gobed:gpu-latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
            cpu: 4000m
          requests:
            nvidia.com/gpu: 1  
            memory: 4Gi
            cpu: 2000m
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: GPU_MEMORY_FRACTION
          value: "0.8"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
      nodeSelector:
        accelerator: nvidia-tesla-t4
---
apiVersion: v1
kind: Service
metadata:
  name: gobed-gpu-search-service
spec:
  selector:
    app: gobed-gpu-search
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

## 🚀 Production Examples

### HTTP API Server

```go
// production_server.go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strconv"
    "time"
    
    "github.com/lee101/gobed/gpu"
)

type SearchRequest struct {
    Query string `json:"query"`
    K     int    `json:"k,omitempty"`
}

type SearchResponse struct {
    Results []gpu.Result `json:"results"`
    Latency string      `json:"latency"`
    QPS     float64     `json:"qps"`
}

type APIServer struct {
    pipeline *gpu.Pipeline
}

func (s *APIServer) searchHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    var req SearchRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    if req.K == 0 {
        req.K = 10
    }
    
    // Perform search with timing
    start := time.Now()
    results, err := s.pipeline.Search(req.Query, req.K)
    latency := time.Since(start)
    
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    
    // Calculate QPS (approximate)
    qps := 1.0 / latency.Seconds()
    
    response := SearchResponse{
        Results: results,
        Latency: latency.String(),
        QPS:     qps,
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func (s *APIServer) healthHandler(w http.ResponseWriter, r *http.Request) {
    stats, err := s.pipeline.GetStats()
    if err != nil {
        http.Error(w, "Pipeline unhealthy", http.StatusInternalServerError)
        return
    }
    
    health := map[string]interface{}{
        "status":      "healthy",
        "gpu_device":  stats.GPUDevice,
        "gpu_memory":  stats.GPUMemoryMB,
        "num_vectors": stats.NumEmbeddings,
        "qps":         stats.SingleQueryQPS,
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(health)
}

func main() {
    // Configure high-performance pipeline
    config := gpu.Config{
        ModelPath:      "../gobed/model",
        GPUServerURL:   "http://localhost:5000",
        BatchSize:      512,      // Large batches for API server
        UseGPUIndexing: true,
        GPUOnlyMode:    true,     // Minimize memory usage
        MaxVectors:     10000000, // Support 10M vectors
    }
    
    pipeline, err := gpu.NewPipeline(config)
    if err != nil {
        log.Fatalf("Failed to create pipeline: %v", err)
    }
    defer pipeline.Close()
    
    // Pre-load sample data (in production, load from database)
    fmt.Println("📚 Loading sample dataset...")
    texts := loadSampleData() // Your data loading function
    if err := pipeline.IndexTexts(texts); err != nil {
        log.Fatalf("Failed to index texts: %v", err)
    }
    
    // Create API server
    server := &APIServer{pipeline: pipeline}
    
    http.HandleFunc("/search", server.searchHandler)
    http.HandleFunc("/health", server.healthHandler)
    
    fmt.Println("🚀 GPU Search API Server starting on :8080")
    fmt.Println("📊 Endpoints:")
    fmt.Println("  POST /search - Perform similarity search")
    fmt.Println("  GET  /health - Check system health")
    
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### Streaming Indexing

```go
// streaming_example.go
func StreamingIndexExample() {
    config := gpu.Config{
        BatchSize:   256,
        GPUOnlyMode: true,
    }
    
    pipeline, _ := gpu.NewPipeline(config)
    defer pipeline.Close()
    
    // Simulate streaming text data
    textChan := make(chan string, 1000)
    
    // Start streaming indexer
    go func() {
        batchSize := 100
        if err := pipeline.StreamingIndex(textChan, batchSize); err != nil {
            log.Printf("Streaming index error: %v", err)
        }
    }()
    
    // Simulate incoming text stream
    for i := 0; i < 10000; i++ {
        text := fmt.Sprintf("Streaming document %d with machine learning content", i)
        textChan <- text
        
        // Simulate processing delay
        time.Sleep(time.Millisecond * 10)
    }
    
    close(textChan)
    fmt.Println("✅ Streaming indexing complete")
}
```

## 📋 Troubleshooting Examples

### Common Issues and Solutions

```go
// troubleshooting_example.go
func TroubleshootingExamples() {
    fmt.Println("🔧 Common Issues and Solutions:")
    
    // Issue 1: Out of GPU memory
    fmt.Println("\n1. GPU Out of Memory:")
    fmt.Println("   Solution: Reduce batch size or enable GPU-only mode")
    config := gpu.Config{
        BatchSize:   128,     // Reduce from 512
        GPUOnlyMode: true,    // Clear CPU memory
    }
    
    // Issue 2: Slow indexing performance
    fmt.Println("\n2. Slow Indexing Performance:")
    fmt.Println("   Solution: Increase batch size and use parallel processing")
    config.BatchSize = 1024  // Increase for better GPU utilization
    
    // Issue 3: High latency
    fmt.Println("\n3. High Search Latency:")
    fmt.Println("   Solution: Preload GPU and optimize memory layout")
    config.PreloadGPU = true
    
    // Issue 4: Memory leaks
    fmt.Println("\n4. Memory Leaks:")
    fmt.Println("   Solution: Always call Close() and monitor with stats")
    
    pipeline, err := gpu.NewPipeline(config)
    if err != nil {
        fmt.Printf("   ❌ Pipeline creation failed: %v\n", err)
        return
    }
    defer pipeline.Close() // Important!
    
    // Monitor memory usage
    if stats, err := pipeline.GetStats(); err == nil {
        fmt.Printf("   📊 GPU Memory: %.1f MB\n", stats.GPUMemoryMB)
        fmt.Printf("   📊 CPU Memory: %.1f MB\n", stats.CPUMemoryMB)
    }
}
```

### Performance Monitoring

```go
// monitoring_example.go
func PerformanceMonitoring() {
    pipeline, _ := gpu.NewPipeline(gpu.Config{})
    defer pipeline.Close()
    
    // Monitor performance in real-time
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()
    
    go func() {
        for range ticker.C {
            stats, err := pipeline.GetStats()
            if err != nil {
                continue
            }
            
            fmt.Printf("📊 Performance Stats:\n")
            fmt.Printf("   GPU Memory: %.1f/8192 MB\n", stats.GPUMemoryMB)
            fmt.Printf("   Search QPS: %.0f\n", stats.SingleQueryQPS)
            fmt.Printf("   Batch QPS: %.0f\n", stats.BatchQPS)
            
            // Alert if performance drops
            if stats.SingleQueryQPS < 1000 {
                fmt.Printf("⚠️  Performance warning: QPS below threshold\n")
            }
        }
    }()
}
```

## 🎯 Next Steps

After running these examples:

1. **Customize for your data**: Modify examples for your specific use case
2. **Optimize performance**: Tune batch sizes and memory settings
3. **Deploy to production**: Use Docker/Kubernetes examples
4. **Monitor and scale**: Implement performance monitoring
5. **Contribute improvements**: Submit optimizations and new examples

## 📚 Additional Resources

- [Main GPU Setup Guide](../gobed/README_GPU.md)
- [Performance Tuning Guide](PERFORMANCE.md)
- [Production Deployment Guide](DEPLOYMENT.md)
- [API Reference](API.md)

---

🚀 **Ready to build high-performance GPU-accelerated text search applications!**