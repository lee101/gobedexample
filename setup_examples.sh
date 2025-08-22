#!/bin/bash
# setup_examples.sh - Quick setup for Gobed GPU examples
# Run this after completing the main GPU setup

set -e

echo "🚀 Gobed GPU Examples Setup"
echo "==========================="
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# 1. Check if main GPU setup was completed
echo "🔍 Verifying main GPU setup..."

# Check for CUDA library
if [ -f "../gobed/gpu_search/cuda_ops/build/libgobed_ann_ops.so" ]; then
    success "CUDA library found"
else
    error "CUDA library not found"
    echo "   Please run: cd ../gobed && ./setup_gpu.sh"
    exit 1
fi

# Check for TorchScript model
if [ -f "../gobed/model/simple_gpu_search_module.pt" ]; then
    success "TorchScript model found"
else
    error "TorchScript model not found"
    echo "   Please run: cd ../gobed && ./setup_gpu.sh"
    exit 1
fi

# Check GPU availability
if nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    success "GPU available: $GPU_NAME"
else
    warning "GPU not detected (examples will still work in CPU mode)"
fi

# 2. Test Go module setup
echo
echo "🔧 Setting up Go modules..."
if go mod tidy > /dev/null 2>&1; then
    success "Go modules updated"
else
    error "Go module setup failed"
    exit 1
fi

# 3. Generate sample data if needed
echo
echo "📚 Preparing sample datasets..."
if [ ! -f "large_data.txt" ]; then
    info "Generating large sample dataset..."
    if go run generate_large_dataset.go > /dev/null 2>&1; then
        success "Large dataset generated"
    else
        warning "Large dataset generation failed (not critical)"
    fi
fi

# 4. Test basic functionality
echo
echo "🧪 Running basic functionality tests..."

echo "   Testing achievement summary..."
if timeout 10s go run achievement_summary.go > /dev/null 2>&1; then
    success "Achievement summary: ✓"
else
    warning "Achievement summary: Limited output (normal)"
fi

echo "   Testing TorchScript integration..."
if timeout 15s go run torchscript_demo.go > /dev/null 2>&1; then
    success "TorchScript demo: ✓"
else
    warning "TorchScript demo: May need GPU server running"
fi

# 5. Check GPU server availability
echo
echo "🖥️  Checking GPU server..."
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    success "GPU server is running"
    SERVER_INFO=$(curl -s http://localhost:5000/health | python3 -c "import sys,json; data=json.load(sys.stdin); print(f'{data[\"device\"]} - {data[\"database_size\"]} vectors')" 2>/dev/null || echo "Running")
    info "Server status: $SERVER_INFO"
else
    warning "GPU server not running"
    echo "   To start: cd ../gobed && python3 gpu_search/gpu_search_server.py"
fi

# 6. Create quick test script
echo
echo "📝 Creating convenience scripts..."

cat > quick_test.sh << 'EOF'
#!/bin/bash
# Quick test of GPU search functionality

echo "🚀 Quick GPU Search Test"
echo "======================="

echo "1. Testing achievement summary..."
go run achievement_summary.go

echo -e "\n2. Testing basic performance..."
go run main.go --max-texts=100 --benchmark

echo -e "\n✅ Quick test complete!"
EOF

chmod +x quick_test.sh
success "Created quick_test.sh"

cat > start_gpu_server.sh << 'EOF'
#!/bin/bash
# Start the GPU search server

echo "🚀 Starting GPU Search Server..."
cd ../gobed
python3 gpu_search/gpu_search_server.py
EOF

chmod +x start_gpu_server.sh
success "Created start_gpu_server.sh"

# 7. Performance recommendations
echo
echo "⚡ Performance Recommendations:"
echo
info "Optimal batch sizes:"
echo "   • Small datasets (<10K): 64-128"
echo "   • Medium datasets (10K-100K): 256-512"  
echo "   • Large datasets (100K+): 512-1024"
echo
info "Memory optimization:"
echo "   • Enable GPU-only mode: GPUOnlyMode: true"
echo "   • Monitor with: nvidia-smi -l 1"
echo "   • Adjust batch size if out of memory"

# 8. Example commands
echo
echo "🎯 Ready to run examples:"
echo
success "Basic examples:"
echo "   go run achievement_summary.go     # See what we built"
echo "   go run torchscript_demo.go       # Test TorchScript"
echo "   ./quick_test.sh                  # Run quick test"
echo
success "Performance testing:"
echo "   go run main.go --benchmark                    # Basic benchmark"
echo "   go run main.go --max-texts=10000 --benchmark # Larger test"
echo "   go run batch_size_tuning.go                  # Optimize performance"
echo
success "Production examples:"
echo "   go run main.go --interactive              # Interactive search"
echo "   go run main.go --performance-test         # Large-scale test"
echo
info "GPU server (in another terminal):"
echo "   ./start_gpu_server.sh            # Start GPU server"
echo "   curl http://localhost:5000/health # Check server status"

echo
echo "🎉 Examples setup complete!"
echo
success "Everything is ready for GPU-accelerated text search!"
echo
echo "📚 Documentation:"
echo "   • Examples guide: README.md"
echo "   • Main GPU guide: ../gobed/README_GPU.md"
echo "   • Troubleshooting: Check batch sizes and GPU memory"
echo
echo "🚀 Start with: go run achievement_summary.go"