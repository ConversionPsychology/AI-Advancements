# Tiny Modular Transformer

A lightweight and modular transformer architecture implemented from scratch in PyTorch. Designed for education, experimentation, and extension.

## 🚀 Features
- <500 lines of core model code (Minimal implementation)
- Fully Modular: tokenization, attention, FFN, embeddings, and layer norm are all swappable (plug-and-play)
- Includes multiple experiment runners: baseline, no layer norm, rotary embeddings
- Interactive Streamlit sandbox for hands-on testing and output visualization 

## Project Structure
`tiny_modular_transformer/`      # Core transformer components
`experiments/`                   # Prebuilt experiments and ablations
`sandbox_app/`                   # Interactive UI (Streamlit)

## Quickstart

### Install Dependencies
```bash
pip install -r requirements.txt
Streamline run sandbox_app/app.py
```

### Run the Interactive Sandbox
```bash
streamlit run sandbox_app/app.py
```

### Try Prebuilt Experiments
```bash
python experiments/run_baseline.py
python experiments/run_no_layernorm.py
python experiments/run_rotary_embeddings.py
```

## Learning Goals
- Understand how transformers actually work internally
- Modify and extend transformer internals
- Visualize architectural decisions in real time

## Future Ideas
- Add FlashAttention / Linformer
- Add memory & long context support
- Export models to ONNX / TorchScript

## License
MIT License

Copyright (c) 2025 ConversionPsychology
│   
│   Permission is hereby granted, free of charge, to any person obtaining a copy
│   of this software and associated documentation files (the "Software"), to deal
│   in the Software without restriction, including without limitation the rights
│   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
│   copies of the Software, and to permit persons to whom the Software is
│   furnished to do so, subject to the following conditions:
│   
│   The above copyright notice and this permission notice shall be included in all
│   copies or substantial portions of the Software.
│   
│   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
│   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
│   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
│   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
│   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
│   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
│   SOFTWARE.

## 📄 .gitignore (recommended)
```
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/

# Environment
.env
.venv/

# Mac
.DS_Store

# Jupyter/Streamlit
.ipynb_checkpoints/
streamlit_app.log

# VSCode or PyCharm
.vscode/
.idea/
```
## Contributing
We welcome contributions from the community! Here's how to get started:

### Local Setup
1. Fork this repository and clone your fork
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the sandbox app or experiment files to verify your setup works

### Ways to Contribute
- Add or experiment with new attention mechanisms (e.g. FlashAttention)
- Improve visual tools in the sandbox (e.g. attention maps)
- Optimize training speed or memory use
- Add language modeling or sequence classification examples
- Report issues, bugs, or suggest ideas via GitHub Issues

### Code Guidelines
- Keep modules clean and small
- Add comments or docstrings for new modules/functions
- Follow PEP8 standards where possible

### Submitting Pull Requests
- Create a new branch: `git checkout -b my-feature`
- Commit changes: `git commit -m "Add custom attention block"`
- Push to your fork: `git push origin my-feature`
- Open a pull request with a clear title and description

Thanks for helping us build an awesome open-source learning tool! 🙏

