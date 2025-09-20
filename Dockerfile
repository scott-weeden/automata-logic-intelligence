# scott-weeden/latex - Comprehensive LaTeX Build Environment
# For Automata Theory Course Documentation and ARC Competition

FROM texlive/texlive:latest-full

LABEL maintainer="Scott Weeden"
LABEL description="LaTeX compilation environment for automata theory course materials and ARC competition documentation"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    make \
    graphviz \
    imagemagick \
    pandoc \
    ghostscript \
    nodejs \
    npm \
    fonts-liberation \
    fonts-dejavu \
    fonts-noto \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for document processing
RUN pip3 install --no-cache-dir \
    matplotlib \
    numpy \
    pandas \
    sympy \
    networkx \
    pygments \
    pyyaml \
    jinja2 \
    pillow \
    beautifulsoup4 \
    lxml

# Install Node.js packages for slide generation
RUN npm install -g \
    reveal-md \
    marp-cli \
    @mermaid-js/mermaid-cli

# Create working directories
RUN mkdir -p /workspace/docs \
    /workspace/output \
    /workspace/templates \
    /workspace/figures \
    /workspace/scripts \
    /workspace/styles

# Copy LaTeX templates and styles
COPY latex-templates/ /workspace/templates/
COPY latex-styles/ /workspace/styles/

# Copy build scripts
COPY scripts/build.sh /workspace/scripts/
COPY scripts/compile.py /workspace/scripts/
COPY scripts/generate-slides.py /workspace/scripts/
COPY scripts/extract-figures.py /workspace/scripts/

# Make scripts executable
RUN chmod +x /workspace/scripts/*.sh

# Set up LaTeX packages specifically for CS theory documentation
RUN tlmgr update --self && \
    tlmgr install \
    algorithm2e \
    algorithmicx \
    algpseudocode \
    listings \
    minted \
    tikz-qtree \
    tikz-dependency \
    forest \
    automata \
    complexity \
    computational-complexity \
    proof \
    stmaryrd \
    mathpartir \
    ebproof \
    bussproofs \
    prftree \
    turnstile \
    semantic \
    syntax \
    qtree \
    tree-dvips

# Configure ImageMagick policy for PDF processing
RUN sed -i 's/<policy domain="coder" rights="none" pattern="PDF" \/>/<policy domain="coder" rights="read|write" pattern="PDF" \/>/g' /etc/ImageMagick-6/policy.xml

# Set environment variables
ENV TEXMFHOME=/workspace/texmf
ENV PATH="/workspace/scripts:${PATH}"
ENV PYTHONPATH="/workspace/scripts:${PYTHONPATH}"

# Create custom build script
RUN cat > /workspace/scripts/renovate-build.sh << 'EOF'
#!/bin/bash
set -e

# Parse arguments
INPUT_FILE=$1
OUTPUT_DIR=${2:-/workspace/output}
BUILD_TYPE=${3:-full}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting LaTeX compilation...${NC}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to compile LaTeX
compile_latex() {
    local file=$1
    local basename=$(basename "$file" .tex)
    
    echo -e "${YELLOW}Compiling: $file${NC}"
    
    # First pass
    pdflatex -interaction=nonstopmode -output-directory="$OUTPUT_DIR" "$file"
    
    # Bibliography if exists
    if grep -q "\\bibliography" "$file"; then
        bibtex "$OUTPUT_DIR/$basename"
    fi
    
    # Makeindex if exists
    if grep -q "\\makeindex" "$file"; then
        makeindex "$OUTPUT_DIR/$basename.idx"
    fi
    
    # Second and third pass for references
    pdflatex -interaction=nonstopmode -output-directory="$OUTPUT_DIR" "$file"
    pdflatex -interaction=nonstopmode -output-directory="$OUTPUT_DIR" "$file"
    
    echo -e "${GREEN}Successfully compiled: $basename.pdf${NC}"
}

# Function to generate slides
generate_slides() {
    local file=$1
    local basename=$(basename "$file" .md)
    
    echo -e "${YELLOW}Generating slides from: $file${NC}"
    marp "$file" -o "$OUTPUT_DIR/$basename-slides.pdf" --theme-set /workspace/styles/
    echo -e "${GREEN}Slides generated: $basename-slides.pdf${NC}"
}

# Main compilation logic
case "$BUILD_TYPE" in
    full)
        # Compile all tex files
        find /workspace/docs -name "*.tex" -type f | while read file; do
            compile_latex "$file"
        done
        
        # Generate slides from markdown
        find /workspace/docs -name "*.md" -type f | while read file; do
            generate_slides "$file"
        done
        ;;
    
    single)
        # Compile single file
        if [[ "$INPUT_FILE" == *.tex ]]; then
            compile_latex "$INPUT_FILE"
        elif [[ "$INPUT_FILE" == *.md ]]; then
            generate_slides "$INPUT_FILE"
        else
            echo -e "${RED}Unsupported file type: $INPUT_FILE${NC}"
            exit 1
        fi
        ;;
    
    exercises)
        # Compile only exercise documents
        find /workspace/docs -path "*/exercises/*.tex" -type f | while read file; do
            compile_latex "$file"
        done
        ;;
    
    arc)
        # Compile ARC competition documents
        find /workspace/docs -path "*/arc/*.tex" -type f | while read file; do
            compile_latex "$file"
        done
        ;;
    
    *)
        echo -e "${RED}Unknown build type: $BUILD_TYPE${NC}"
        echo "Available types: full, single, exercises, arc"
        exit 1
        ;;
esac

echo -e "${GREEN}Build complete!${NC}"
ls -la "$OUTPUT_DIR"
EOF

RUN chmod +x /workspace/scripts/renovate-build.sh

# Default working directory
WORKDIR /workspace

# Entry point
ENTRYPOINT ["/workspace/scripts/renovate-build.sh"]

# Default command
CMD ["full"]
