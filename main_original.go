// main_original.go - Test with original sequential approach
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/lee101/gobed/gpu"
)

func main() {
	// Command line flags
	var (
		gpuServer   = flag.String("gpu-server", "http://localhost:5000", "GPU server URL")
		modelPath   = flag.String("model", "../gobed/model", "Path to model directory")
		dataFile    = flag.String("data", "large_data.txt", "Text file to index")
		interactive = flag.Bool("interactive", false, "Interactive search mode")
		benchmark   = flag.Bool("benchmark", false, "Run performance benchmark")
		startServer = flag.Bool("start-server", false, "Start GPU server automatically")
		batchSize   = flag.Int("batch-size", 256, "Batch size for GPU processing")
		maxTexts    = flag.Int("max-texts", 0, "Maximum number of texts to index (0 = all)")
	)
	flag.Parse()

	// Start GPU server if requested
	if *startServer {
		if err := startGPUServer(); err != nil {
			log.Printf("Warning: Could not start GPU server: %v", err)
			log.Println("Please start it manually with: python3 ../gobed/gpu_search/gpu_search_server.py")
		} else {
			log.Println("✅ GPU server started")
			time.Sleep(3 * time.Second) // Wait for server to initialize
		}
	}

	// Initialize GPU pipeline - ORIGINAL settings
	config := gpu.Config{
		ModelPath:      *modelPath,
		GPUServerURL:   *gpuServer,
		BatchSize:      *batchSize, // Original batch size
		UseGPUIndexing: true,
		PreloadGPU:     false,      // Original setting
		MaxVectors:     1000000,
		GPUOnlyMode:    true,
	}

	pipeline, err := gpu.NewPipeline(config)
	if err != nil {
		log.Fatalf("Failed to create GPU pipeline: %v", err)
	}

	log.Printf("✅ Original GPU Pipeline initialized")
	log.Printf("   Batch size: %d (original)", *batchSize)
	log.Printf("   GPU-only mode: %t", config.GPUOnlyMode)
	log.Printf("   Preload GPU: %t", config.PreloadGPU)

	// Load or create sample data
	texts, err := loadTexts(*dataFile)
	if err != nil {
		log.Printf("Could not load data file %s: %v", *dataFile, err)
		log.Println("Creating sample data...")
		texts = createSampleData()
		saveTexts(*dataFile, texts)
	}

	log.Printf("📚 Loaded %d texts for indexing", len(texts))
	
	// Limit texts if requested
	if *maxTexts > 0 && len(texts) > *maxTexts {
		texts = texts[:*maxTexts]
		log.Printf("📝 Limited to %d texts for testing", len(texts))
	}

	// Index texts on GPU - ORIGINAL SEQUENTIAL approach
	log.Println("🚀 Starting ORIGINAL sequential indexing...")
	start := time.Now()
	
	if err := pipeline.IndexTexts(texts); err != nil {
		log.Fatalf("Failed to index texts: %v", err)
	}
	
	indexTime := time.Since(start)
	throughput := float64(len(texts)) / indexTime.Seconds()
	
	log.Printf("✅ Original sequential indexing complete!")
	log.Printf("   Total texts: %d", len(texts))
	log.Printf("   Total time: %v", indexTime)
	log.Printf("   Throughput: %.0f texts/sec", throughput)

	// Get pipeline statistics
	stats, err := pipeline.GetStats()
	if err == nil {
		log.Printf("📊 Pipeline Stats:")
		log.Printf("   Texts: %d", stats.NumTexts)
		log.Printf("   Embeddings: %d", stats.NumEmbeddings)
		log.Printf("   GPU Device: %s", stats.GPUDevice)
		log.Printf("   GPU Memory: %.1f MB", stats.GPUMemoryMB)
		if stats.CPUMemoryMB > 0 {
			log.Printf("   CPU Memory: %.1f MB", stats.CPUMemoryMB)
		} else {
			log.Printf("   CPU Memory: 0 MB (GPU-only mode)")
		}
		log.Printf("   Single Query: %.2fms (%.0f QPS)", stats.SingleQueryMS, stats.SingleQueryQPS)
		log.Printf("   Batch QPS: %.0f", stats.BatchQPS)
	}
}

func startGPUServer() error {
	cmd := exec.Command("python3", "../gobed/gpu_search/gpu_search_server.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Start()
}

func loadTexts(filename string) ([]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var texts []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text != "" {
			texts = append(texts, text)
		}
	}

	return texts, scanner.Err()
}

func saveTexts(filename string, texts []string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for _, text := range texts {
		fmt.Fprintln(file, text)
	}
	return nil
}

func createSampleData() []string {
	return []string{
		// Technology
		"Artificial intelligence is transforming how we interact with technology",
		"Machine learning models can now understand and generate human language",
		"Deep learning has revolutionized computer vision and image recognition",
		"Neural networks are inspired by the structure of the human brain",
		"GPUs accelerate machine learning computations by orders of magnitude",
		"Transformer models have become the foundation of modern NLP",
		"BERT and GPT models have set new benchmarks in language understanding",
		"Computer vision can now detect objects with superhuman accuracy",
		"Reinforcement learning enables AI to master complex games and tasks",
		"Edge computing brings AI inference closer to data sources",
	}
}