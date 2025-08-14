package main

import (
	"fmt"
	"log"
	"math"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

// Calculate Euclidean distance between two embeddings
func euclideanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// Calculate Manhattan distance between two embeddings
func manhattanDistance(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := range a {
		sum += float32(math.Abs(float64(a[i] - b[i])))
	}
	return sum
}

func main() {
	fmt.Println("🚀 Gobed Distance Metrics Example")
	fmt.Println("==================================")
	
	// Load the embedding model
	fmt.Print("Loading model... ")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("✓ (%.2fs)\n\n", time.Since(start).Seconds())

	// Define test texts - grouped by similarity
	relatedTexts := []struct {
		category string
		texts    []string
	}{
		{
			category: "🤖 Machine Learning",
			texts: []string{
				"Machine learning is fascinating.",
				"Deep learning models are powerful.",
				"Neural networks process information.",
				"Artificial intelligence will change the world.",
			},
		},
		{
			category: "💻 Programming",
			texts: []string{
				"Python is a programming language.",
				"JavaScript runs in browsers.",
				"Code should be readable.",
			},
		},
		{
			category: "🌤️ Daily Life",
			texts: []string{
				"The weather is nice today.",
				"Good morning everyone",
				"Pizza tastes delicious.",
			},
		},
		{
			category: "🌳 Nature",
			texts: []string{
				"Trees grow tall in the forest.",
				"Birds are singing beautifully.",
			},
		},
	}

	// Encode all texts
	fmt.Println("📝 Encoding texts and calculating distances...")
	fmt.Println()
	
	type textEmbedding struct {
		text     string
		category string
		embedding []float32
	}
	
	var allTexts []textEmbedding
	for _, group := range relatedTexts {
		for _, text := range group.texts {
			emb, err := model.Encode(text)
			if err != nil {
				fmt.Printf("Warning: couldn't encode '%s': %v\n", text, err)
				continue
			}
			allTexts = append(allTexts, textEmbedding{
				text:     text,
				category: group.category,
				embedding: emb,
			})
		}
	}

	// 1. Show distances within same category (should be smaller)
	fmt.Println("📊 DISTANCES WITHIN SAME CATEGORY (Related Texts)")
	fmt.Println(strings.Repeat("=", 51))
	
	for _, group := range relatedTexts {
		if len(group.texts) < 2 {
			continue
		}
		
		fmt.Printf("\n%s %s:\n", group.category, group.category[:4])
		
		// Get embeddings for this category
		var categoryEmbs []textEmbedding
		for _, te := range allTexts {
			if te.category == group.category {
				categoryEmbs = append(categoryEmbs, te)
			}
		}
		
		// Calculate pairwise distances within category
		for i := 0; i < len(categoryEmbs)-1; i++ {
			for j := i + 1; j < len(categoryEmbs); j++ {
				sim, _ := model.Similarity(categoryEmbs[i].text, categoryEmbs[j].text)
				eucDist := euclideanDistance(categoryEmbs[i].embedding, categoryEmbs[j].embedding)
				manDist := manhattanDistance(categoryEmbs[i].embedding, categoryEmbs[j].embedding)
				
				fmt.Printf("  '%s'\n  '%s'\n", 
					categoryEmbs[i].text, categoryEmbs[j].text)
				fmt.Printf("    → Cosine Similarity: %.4f (↑ higher = more similar)\n", sim)
				fmt.Printf("    → Euclidean Distance: %.2f (↓ lower = more similar)\n", eucDist)
				fmt.Printf("    → Manhattan Distance: %.2f (↓ lower = more similar)\n", manDist)
				fmt.Println()
			}
		}
	}

	// 2. Show distances between different categories (should be larger)
	fmt.Println("📊 DISTANCES BETWEEN DIFFERENT CATEGORIES (Unrelated Texts)")
	fmt.Println(strings.Repeat("=", 59))
	
	// Compare texts from different categories
	comparisons := []struct {
		text1 string
		cat1  string
		text2 string
		cat2  string
	}{
		{
			"Machine learning is fascinating.",
			"🤖 ML",
			"Pizza tastes delicious.",
			"🍕 Food",
		},
		{
			"Neural networks process information.",
			"🤖 ML",
			"The weather is nice today.",
			"🌤️ Daily",
		},
		{
			"Python is a programming language.",
			"💻 Code",
			"Birds are singing beautifully.",
			"🌳 Nature",
		},
		{
			"Deep learning models are powerful.",
			"🤖 ML",
			"Good morning everyone",
			"👋 Greeting",
		},
	}
	
	fmt.Println()
	for _, comp := range comparisons {
		// Find embeddings
		var emb1, emb2 []float32
		for _, te := range allTexts {
			if te.text == comp.text1 {
				emb1 = te.embedding
			}
			if te.text == comp.text2 {
				emb2 = te.embedding
			}
		}
		
		if len(emb1) > 0 && len(emb2) > 0 {
			sim, _ := model.Similarity(comp.text1, comp.text2)
			eucDist := euclideanDistance(emb1, emb2)
			manDist := manhattanDistance(emb1, emb2)
			
			fmt.Printf("%s '%s'\n%s '%s'\n", 
				comp.cat1, comp.text1, comp.cat2, comp.text2)
			fmt.Printf("    → Cosine Similarity: %.4f (↑ higher = more similar)\n", sim)
			fmt.Printf("    → Euclidean Distance: %.2f (↓ lower = more similar)\n", eucDist)
			fmt.Printf("    → Manhattan Distance: %.2f (↓ lower = more similar)\n", manDist)
			fmt.Println()
		}
	}

	// 3. Summary statistics
	fmt.Println("📈 SUMMARY STATISTICS")
	fmt.Println(strings.Repeat("=", 21))
	
	var relatedSims, unrelatedSims []float32
	var relatedEucDists, unrelatedEucDists []float32
	
	// Calculate all pairwise metrics
	for i := 0; i < len(allTexts); i++ {
		for j := i + 1; j < len(allTexts); j++ {
			sim, _ := model.Similarity(allTexts[i].text, allTexts[j].text)
			eucDist := euclideanDistance(allTexts[i].embedding, allTexts[j].embedding)
			
			if allTexts[i].category == allTexts[j].category {
				relatedSims = append(relatedSims, sim)
				relatedEucDists = append(relatedEucDists, eucDist)
			} else {
				unrelatedSims = append(unrelatedSims, sim)
				unrelatedEucDists = append(unrelatedEucDists, eucDist)
			}
		}
	}
	
	// Calculate averages
	avgRelSim := average(relatedSims)
	avgUnrelSim := average(unrelatedSims)
	avgRelEuc := average(relatedEucDists)
	avgUnrelEuc := average(unrelatedEucDists)
	
	fmt.Printf("\n📊 Average Cosine Similarity:\n")
	fmt.Printf("   Related texts:   %.4f (higher is better)\n", avgRelSim)
	fmt.Printf("   Unrelated texts: %.4f\n", avgUnrelSim)
	fmt.Printf("   Difference:      %.4f (larger gap is better)\n", avgRelSim - avgUnrelSim)
	
	fmt.Printf("\n📏 Average Euclidean Distance:\n")
	fmt.Printf("   Related texts:   %.2f (lower is better)\n", avgRelEuc)
	fmt.Printf("   Unrelated texts: %.2f\n", avgUnrelEuc)
	fmt.Printf("   Difference:      %.2f (larger gap is better)\n", avgUnrelEuc - avgRelEuc)
	
	// Show that the model can distinguish
	fmt.Println("\n✅ Model Quality Check:")
	if avgRelSim > avgUnrelSim && avgRelEuc < avgUnrelEuc {
		fmt.Println("   ✓ Model correctly identifies related texts as more similar")
		fmt.Println("   ✓ Related texts have higher similarity and lower distance")
		fmt.Printf("   ✓ Discrimination ratio: %.2fx better similarity for related texts\n", 
			avgRelSim/avgUnrelSim)
	} else {
		fmt.Println("   ⚠️ Unexpected results - check model or texts")
	}
	
	fmt.Println("\n🎉 Done!")
}

func average(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	var sum float32
	for _, v := range values {
		sum += v
	}
	return sum / float32(len(values))
}