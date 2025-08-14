package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🚀 Gobed Similarity Calculator")
	fmt.Println("================================")
	fmt.Println()

	// Load the model
	fmt.Print("Loading model... ")
	model, err := gobed.LoadModel()
	if err != nil {
		fmt.Printf("\n❌ Failed to load model: %v\n", err)
		fmt.Println("\nMake sure you have the model files:")
		fmt.Println("  1. Run: git clone https://github.com/lee101/gobed ../gobed")
		fmt.Println("  2. Run: cd ../gobed && ./setup.sh")
		fmt.Println("  3. Run: ln -s ../gobed/model model")
		os.Exit(1)
	}
	fmt.Println("✓")
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)

	for {
		// Get first text
		fmt.Print("Enter first text (or 'quit' to exit): ")
		text1, err := reader.ReadString('\n')
		if err != nil {
			if err.Error() == "EOF" {
				fmt.Println("\n👋 Goodbye!")
				break
			}
			fmt.Printf("❌ Error reading input: %v\n", err)
			continue
		}
		text1 = strings.TrimSpace(text1)

		if text1 == "quit" || text1 == "exit" || text1 == "q" {
			fmt.Println("\n👋 Goodbye!")
			break
		}

		// Get second text
		fmt.Print("Enter second text: ")
		text2, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("❌ Error reading input: %v\n", err)
			continue
		}
		text2 = strings.TrimSpace(text2)

		if text2 == "quit" || text2 == "exit" || text2 == "q" {
			fmt.Println("\n👋 Goodbye!")
			break
		}

		// Calculate similarity
		fmt.Println("\n📊 Calculating...")
		
		// First check if texts are in reference tokens
		similarity, err := model.Similarity(text1, text2)
		if err != nil {
			// If not in reference tokens, we need to add them
			// For now, we'll show available texts
			fmt.Printf("\n⚠️  Error: %v\n", err)
			fmt.Println("\nNote: The current model only supports pre-tokenized texts.")
			fmt.Println("Available example texts include:")
			
			availableTexts := model.GetAvailableTexts()
			for i, text := range availableTexts {
				if i >= 10 {
					fmt.Printf("  ... and %d more\n", len(availableTexts)-10)
					break
				}
				fmt.Printf("  • %s\n", text)
			}
			fmt.Println("\nTry using some of these texts for testing.")
		} else {
			// Display results
			fmt.Println("\n" + strings.Repeat("=", 60))
			fmt.Printf("📝 Text 1: \"%s\"\n", text1)
			fmt.Printf("📝 Text 2: \"%s\"\n", text2)
			fmt.Println(strings.Repeat("-", 60))
			
			// Similarity score with interpretation
			fmt.Printf("✨ Cosine Similarity: %.4f\n", similarity)
			
			// Interpret the score
			if similarity > 0.8 {
				fmt.Println("   → Nearly identical meaning! 🎯")
			} else if similarity > 0.5 {
				fmt.Println("   → Very similar texts 🔥")
			} else if similarity > 0.3 {
				fmt.Println("   → Moderately similar ✓")
			} else if similarity > 0.1 {
				fmt.Println("   → Somewhat related 🔗")
			} else if similarity > 0 {
				fmt.Println("   → Weakly related 💭")
			} else {
				fmt.Println("   → Unrelated or opposite meanings ❄️")
			}
			
			// Distance metric
			distance := 1.0 - similarity
			fmt.Printf("📏 Distance: %.4f (1 - similarity)\n", distance)
			
			// Percentage similarity
			percentage := similarity * 100
			if percentage < 0 {
				percentage = 0
			}
			fmt.Printf("📊 Similarity percentage: %.1f%%\n", percentage)
			
			fmt.Println(strings.Repeat("=", 60))
		}
		
		fmt.Println()
	}
}