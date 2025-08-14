#!/bin/bash

echo "Building similarity calculator..."
go build -o similarity similarity.go

if [ $? -eq 0 ]; then
    echo "Running..."
    echo ""
    ./similarity
else
    echo "Build failed. Please check your Go installation."
fi