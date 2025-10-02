# 🔍 Personal Fact-Checker Agent

A Python-based AI system that automatically verifies claims by searching trusted sources (Wikipedia, arXiv, government datasets, news feeds) and provides evidence-based verdicts with confidence scores.

## 🌟 Features

- **Multi-Source Evidence Retrieval**: Searches Wikipedia, arXiv, and other trusted sources
- **Semantic Search**: Uses sentence-transformers and FAISS for intelligent evidence matching
- **NLI-Based Verification**: Employs RoBERTa-MNLI model to classify evidence as supporting/refuting/neutral
- **Confidence Scoring**: Aggregates evidence strength and source credibility into confidence scores
- **Multiple Interfaces**: Both web UI (Gradio) and command-line interface
- **Detailed Reasoning**: Optional step-by-step explanation of the verification process

## 🚀 Quick Start

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the demo** (recommended first step):
```bash
python demo.py
```

### Usage Options

#### 🌐 Web Interface (Recommended)
```bash
# Start the web interface
python main.py --web

# Custom port and public sharing
python main.py --web --port 8080 --share
```

#### 💻 Command Line Interface
```bash
# Check a single claim
python main.py --cli --claim "The Earth is round"

# Interactive mode
python main.py --cli --interactive

# With detailed steps and JSON output
python main.py --cli --claim "COVID vaccines are effective" --steps --json
```

## 📁 Project Structure

```
personal-fact-checker/
├── main.py                 # Main entry point
├── demo.py                 # Demo script with sample claims
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── data_sources/           # Data source connectors
│   ├── __init__.py
│   ├── base_connector.py   # Abstract base class
│   ├── wikipedia_connector.py
│   └── arxiv_connector.py
├── retrieval/              # Evidence retrieval and search
│   ├── __init__.py
│   ├── embedding_manager.py    # Sentence embeddings
│   ├── semantic_search.py      # FAISS-based search
│   └── evidence_retriever.py   # Main retrieval orchestrator
├── verification/           # Claim verification logic
│   ├── __init__.py
│   ├── nli_classifier.py       # RoBERTa-MNLI inference
│   ├── verdict_aggregator.py   # Evidence aggregation
│   └── fact_checker.py         # Main fact-checker
└── interfaces/             # User interfaces
    ├── __init__.py
    ├── gradio_app.py           # Web interface
    └── cli_interface.py        # Command-line interface
```

## 🔧 How It Works

1. **Evidence Retrieval**: Searches multiple sources (Wikipedia, arXiv) for relevant information
2. **Semantic Ranking**: Uses sentence-transformers to rank evidence by semantic similarity to the claim
3. **NLI Classification**: RoBERTa-MNLI model classifies each evidence piece as:
   - `ENTAILMENT` (supports the claim)
   - `CONTRADICTION` (refutes the claim)  
   - `NEUTRAL` (unrelated/insufficient)
4. **Verdict Aggregation**: Combines evidence strength and source credibility to determine:
   - **Likely True** (high supporting evidence)
   - **Likely False** (high refuting evidence)
   - **Unverified** (insufficient or conflicting evidence)

## 🎯 Example Results

**Claim**: "The Earth is round"
- **Verdict**: Likely True (85% confidence)
- **Evidence**: 3 supporting sources from Wikipedia
- **Reasoning**: Strong supporting evidence found from reliable sources

**Claim**: "The Great Wall of China is visible from space"
- **Verdict**: Likely False (78% confidence)  
- **Evidence**: 2 refuting sources explaining this is a myth
- **Reasoning**: Multiple reliable sources contradict this claim

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Model settings
embedding_model = "all-MiniLM-L6-v2"
nli_model = "roberta-large-mnli"
device = "cpu"  # Change to "cuda" for GPU

# Verdict thresholds
true_threshold = 0.7
false_threshold = 0.7

# Retrieval settings
max_results_per_source = 5
similarity_threshold = 0.3
```

## 🔌 Extending the System

### Adding New Data Sources

1. Create a new connector in `data_sources/`:
```python
from .base_connector import BaseConnector, EvidenceItem

class MyConnector(BaseConnector):
    async def search(self, query: str, max_results: int = 5) -> List[EvidenceItem]:
        # Implement your search logic
        pass
```

2. Register it in `retrieval/evidence_retriever.py`:
```python
self.connectors["my_source"] = MyConnector()
```

### Customizing Verification Logic

- Modify `verification/verdict_aggregator.py` to change how evidence is weighted
- Adjust confidence thresholds in `config.py`
- Add custom NLI models in `verification/nli_classifier.py`

## 🛠️ Development

### Running Tests
```bash
# TODO: Add pytest tests
pytest tests/
```

### Code Quality
```bash
# Format code
black .

# Lint code  
flake8 .
```

## 📋 TODO / Future Enhancements

- [ ] Add CDC and data.gov connectors
- [ ] Implement news feed RSS parsing
- [ ] Add caching for API responses
- [ ] Create comprehensive test suite
- [ ] Add support for multiple languages
- [ ] Implement user feedback system
- [ ] Add batch processing capabilities
- [ ] Create API endpoint for external integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **Sentence Transformers** for semantic embeddings
- **FAISS** for efficient similarity search
- **Gradio** for the web interface
- **Wikipedia** and **arXiv** for providing open access to knowledge

## 📞 Support

If you encounter issues or have questions:

1. Check the demo script works: `python demo.py`
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Check system status in interactive CLI: `python main.py --cli --interactive` → type `status`

For common issues:
- **Memory errors**: Reduce `max_results_per_source` in config
- **Slow performance**: Check if GPU is available and set `device = "cuda"`
- **Network errors**: Verify internet connection and API availability



