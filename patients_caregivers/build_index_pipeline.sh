#!/bin/bash

set -e
set -o pipefail

echo "📦 Starting full data pipeline..."

# Step 0: Handle existing data directory
if [ -d "data" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mv data "data_old_$TIMESTAMP"
    echo "🔁 Moved existing 'data/' to 'data_old_$TIMESTAMP'"
fi

# Step 1: Create new data directory
mkdir -p data
echo "📁 Created new 'data/' directory"

# Step 2: Move files from feed_docs/ into data/ if feed_docs exists
if [ -d "feed_docs" ]; then
    mv feed_docs/* data/ 2>/dev/null || true
    echo "📂 Moved documents from 'feed_docs/' to 'data/'"
else
    echo "⚠️ No 'feed_docs/' directory found; skipping file move."
fi

# Step 3: Crawl URLs into WARC
echo "🌐 [1/2] Crawling with wget..."
wget --input-file all_urls.txt \
     --recursive \
     --level 2 \
     --delete-after \
     --no-directories \
     --warc-file data/crawl

# Step 4: Preprocess WARC with Docker
echo "⚙️  [2/2] Running Docker preprocessing..."
sudo docker run --rm \
    -v "$(pwd)/data":/data \
    opencode.it4i.eu:5050/openwebsearcheu-public/preprocessing-pipeline \
    /data/crawl.warc.gz /data/metadata.parquet

# Step 5: Index and persist to disk
echo "Note: After crawling for all languages, run index_save_to_disk.py to build the unified index."
echo "🧠 Building unified index from all language folders..."
python index_save_to_disk.py

echo "✅ All steps completed successfully!"
