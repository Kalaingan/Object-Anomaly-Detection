#!/bin/bash

# Make the script exit on error
set -e

# Define help function
function show_help {
    echo "Usage: ./run.sh [OPTION]"
    echo "Run different components of the anomaly detection application."
    echo ""
    echo "Options:"
    echo "  train       Train the model using the provided dataset"
    echo "  deploy      Run the Streamlit web application"
    echo "  test IMAGE  Test the model on a single image"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train"
    echo "  ./run.sh deploy"
    echo "  ./run.sh test path/to/image.jpg"
}

# Check if any arguments are provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

# Process arguments
case "$1" in
    train)
        echo "Training the model..."
        python app.py
        ;;
    deploy)
        echo "Starting Streamlit application..."
        streamlit run streamlit_app.py
        ;;
    test)
        if [ $# -lt 2 ]; then
            echo "Error: No image path provided for testing."
            echo "Usage: ./run.sh test path/to/image.jpg"
            exit 1
        fi
        echo "Testing model on image: $2"
        python test_model.py "$2"
        ;;
    help)
        show_help
        ;;
    *)
        echo "Error: Unknown option '$1'"
        show_help
        exit 1
        ;;
esac 
