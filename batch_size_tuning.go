// batch_size_tuning.go - Find optimal batch sizes for GPU processing
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/lee101/gobed/gpu"
)

func main() {
	var (
		modelPath = flag.String("model", "../gobed/model", "Path to model directory")
		dataFile  = flag.String("data", "large_data.txt", "Text file to index")
		maxTexts  = flag.Int("max-texts", 50000, "Maximum number of texts to test")
	)
	flag.Parse()

	fmt.Println("🚀 GPU Batch Size Optimization Suite")
	fmt.Println("=" + strings.Repeat("=", 50))

	// Test different batch sizes
	batchSizes := []int{16, 32, 64, 128, 256, 512, 1024, 2048}
	results := make(map[int]BenchmarkResult)

	for _, batchSize := range batchSizes {
		fmt.Printf("\n📊 Testing batch size: %d\n", batchSize)
		result := testBatchSize(batchSize, *modelPath, *dataFile, *maxTexts)
		results[batchSize] = result
		
		fmt.Printf("   Indexing: %.0f texts/sec\n", result.IndexingTPS)
		fmt.Printf("   Search latency: %.2fms\n", result.SearchLatencyMs)
		fmt.Printf("   Search QPS: %.0f\n", result.SearchQPS)
		fmt.Printf("   Batch QPS: %.0f\n", result.BatchQPS)
		fmt.Printf("   GPU Memory: %.1f MB\n", result.MemoryMB)
	}

	// Summary report
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("📈 BATCH SIZE OPTIMIZATION RESULTS")
	fmt.Println(strings.Repeat("=", 60))
	
	fmt.Printf("%-10s | %-12s | %-10s | %-10s | %-10s | %-8s\n",
		"Batch", "Index TPS", "Latency", "Search QPS", "Batch QPS", "Memory")
	fmt.Printf("%-10s | %-12s | %-10s | %-10s | %-10s | %-8s\n",
		"Size", "", "(ms)", "", "", "(MB)")
	fmt.Println(strings.Repeat("-", 75))

	bestIndexing := 0
	bestSearch := 0
	bestBatch := 0
	maxIndexTPS := 0.0
	maxSearchQPS := 0.0
	maxBatchQPS := 0.0

	for _, batchSize := range batchSizes {
		result := results[batchSize]
		fmt.Printf("%-10d | %-12.0f | %-10.2f | %-10.0f | %-10.0f | %-8.1f\n",
			batchSize, result.IndexingTPS, result.SearchLatencyMs,
			result.SearchQPS, result.BatchQPS, result.MemoryMB)

		if result.IndexingTPS > maxIndexTPS {
			maxIndexTPS = result.IndexingTPS
			bestIndexing = batchSize
		}
		if result.SearchQPS > maxSearchQPS {
			maxSearchQPS = result.SearchQPS
			bestSearch = batchSize
		}
		if result.BatchQPS > maxBatchQPS {
			maxBatchQPS = result.BatchQPS
			bestBatch = batchSize
		}
	}

	fmt.Println("\n🏆 OPTIMAL CONFIGURATIONS:")
	fmt.Printf("   Best for indexing: %d (%.0f texts/sec)\n", bestIndexing, maxIndexTPS)
	fmt.Printf("   Best for search: %d (%.0f QPS)\n", bestSearch, maxSearchQPS)
	fmt.Printf("   Best for batch: %d (%.0f QPS)\n", bestBatch, maxBatchQPS)

	// Recommendations
	fmt.Println("\n💡 RECOMMENDATIONS:")
	if bestIndexing == bestSearch && bestSearch == bestBatch {
		fmt.Printf("   Use batch size %d for optimal performance across all operations\n", bestIndexing)
	} else {
		fmt.Printf("   For indexing-heavy workloads: batch size %d\n", bestIndexing)
		fmt.Printf("   For search-heavy workloads: batch size %d\n", bestSearch)
		fmt.Printf("   For batch processing: batch size %d\n", bestBatch)
	}

	// Memory efficiency analysis
	fmt.Println("\n💾 MEMORY EFFICIENCY:")
	for _, batchSize := range batchSizes {
		result := results[batchSize]
		efficiency := result.SearchQPS / result.MemoryMB
		fmt.Printf("   Batch %4d: %.1f QPS per MB\n", batchSize, efficiency)
	}
}

type BenchmarkResult struct {
	BatchSize       int
	IndexingTPS     float64
	SearchLatencyMs float64
	SearchQPS       float64
	BatchQPS        float64
	MemoryMB        float64
}

func testBatchSize(batchSize int, modelPath, dataFile string, maxTexts int) BenchmarkResult {
	// Initialize pipeline
	config := gpu.Config{
		ModelPath:      modelPath,
		GPUServerURL:   "http://localhost:5000",
		BatchSize:      batchSize,
		UseGPUIndexing: true,
		PreloadGPU:     false,
		MaxVectors:     maxTexts,
	}

	pipeline, err := gpu.NewPipeline(config)
	if err != nil {
		log.Printf("Failed to create pipeline: %v", err)
		return BenchmarkResult{}
	}

	// Load texts
	texts, err := loadTexts(dataFile, maxTexts)
	if err != nil {
		log.Printf("Failed to load texts: %v", err)
		return BenchmarkResult{}
	}

	// Benchmark indexing
	start := time.Now()
	err = pipeline.IndexTexts(texts)
	if err != nil {
		log.Printf("Failed to index texts: %v", err)
		return BenchmarkResult{}
	}
	indexTime := time.Since(start)
	indexingTPS := float64(len(texts)) / indexTime.Seconds()

	// Benchmark search
	query := "artificial intelligence and machine learning"
	
	// Warmup
	for i := 0; i < 10; i++ {
		pipeline.Search(query, 10)
	}

	// Single query benchmark
	iterations := 50
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, err = pipeline.Search(query, 10)
		if err != nil {
			log.Printf("Search error: %v", err)
		}
	}
	searchTime := time.Since(start)
	avgLatency := searchTime / time.Duration(iterations)
	searchQPS := float64(iterations) / searchTime.Seconds()

	// Batch benchmark
	queries := []string{
		"artificial intelligence and machine learning",
		"health benefits of exercise",
		"quantum physics and computing",
		"financial investment strategies",
		"climate and environment",
		"database optimization techniques",
		"network security protocols",
		"software development practices",
	}

	batchIterations := 10
	start = time.Now()
	for i := 0; i < batchIterations; i++ {
		_, err = pipeline.BatchSearch(queries, 10)
		if err != nil {
			log.Printf("Batch search error: %v", err)
		}
	}
	batchTime := time.Since(start)
	totalQueries := len(queries) * batchIterations
	batchQPS := float64(totalQueries) / batchTime.Seconds()

	// Get memory usage
	stats, err := pipeline.GetStats()
	memoryMB := 0.0
	if err == nil {
		memoryMB = stats.GPUMemoryMB
	}

	return BenchmarkResult{
		BatchSize:       batchSize,
		IndexingTPS:     indexingTPS,
		SearchLatencyMs: float64(avgLatency.Nanoseconds()) / 1e6,
		SearchQPS:       searchQPS,
		BatchQPS:        batchQPS,
		MemoryMB:        memoryMB,
	}
}

func loadTexts(filename string, maxTexts int) ([]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var texts []string
	
	// Use faster file reading for large files
	cmd := exec.Command("head", "-"+strconv.Itoa(maxTexts), filename)
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			texts = append(texts, line)
		}
	}

	return texts, nil
}