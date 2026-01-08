# Sentiment gRPC Server

A Python gRPC service for multilingual sentiment analysis, currently supporting **Hungarian**, **English** and **Danish**, designed to be consumed by other services (e.g. a Go API) via a stable gRPC contract.

## Features

- gRPC API for sentiment analysis  
- Single and batch text analysis  
- Language-based analyzer routing  
- Efficient model reuse via singleton analyzers  
- Designed for service-to-service communication  