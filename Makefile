# Makefile for Accelera development

.PHONY: help install bin-install test benchmark examples clean lint docs

help:
	@echo "Accelera Development Commands:"
	@echo "  install     - Install package dependencies (pip install)"
	@echo "  bin-install - Install accelera-python binary wrapper"
	@echo "  test        - Run unit tests"
	@echo "  benchmark   - Run performance benchmarks"
	@echo "  examples    - Run example scripts"
	@echo "  clean       - Clean up build artifacts"
	@echo "  lint        - Run code linting"
	@echo "  docs        - Build documentation"

install:
	python3 -m pip install . --break-system-packages

bin-install:
	@echo "🚀 Accelera Binary Installation"
	@echo "==============================="
	@echo "Installing Accelera transparent PyTorch wrapper..."
	@echo
	@command -v python3 >/dev/null 2>&1 || (echo "❌ Error: python3 is required but not found" && exit 1)
	@python3 -c "import torch" >/dev/null 2>&1 || (echo "❌ Error: PyTorch is required but not found" && echo "Please install PyTorch first: pip install torch" && exit 1)
	@echo "✅ Python 3 found: $$(python3 --version)"
	@echo "✅ PyTorch found: $$(python3 -c 'import torch; print(f"PyTorch {torch.__version__}")')"
	@chmod +x bin/accelera-python
	@echo "Creating symlink in $$HOME/.local/bin..."
	@mkdir -p $$HOME/.local/bin
	@rm -f $$HOME/.local/bin/accelera-python
	@ln -s $$(pwd)/bin/accelera-python $$HOME/.local/bin/accelera-python
	@echo "✅ Created symlink: $$HOME/.local/bin/accelera-python -> $$(pwd)/bin/accelera-python"
	@if [ "$$PATH" != "*$$HOME/.local/bin*" ]; then \
		echo "⚠️  $$HOME/.local/bin is not in your PATH"; \
		echo "   Add this line to your ~/.bashrc or ~/.zshrc:"; \
		echo "   export PATH=\"$$HOME/.local/bin:$$PATH\""; \
	fi
	@echo
	@echo "🎉 Binary installation complete!"
	@echo
	@echo "Usage examples:"
	@echo "  accelera-python --accelera-help          # Show help"
	@echo "  accelera-python --accelera-status        # Check status"
	@echo "  accelera-python your_script.py           # Run script with Accelera"
	@echo "  accelera-python --accelera-verbose train.py  # Run with verbose logging"
	@echo
	@echo "Test the installation:"
	@echo "  accelera-python -c \"import torch; print('✅ Accelera ready!')\""

test:
	python -m pytest tests/ -v

test-cuda:
	@echo "Testing CUDA availability..."
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@echo "Running CUDA-dependent tests..."
	python tests/test_accelera.py

benchmark:
	python examples/benchmark.py

benchmark-custom:
	@echo "Running custom benchmark (2000x1500 @ 1500x2500)..."
	python examples/benchmark.py --custom-size 2000 1500 2500

examples:
	@echo "Running basic example..."
	python examples/basic_usage.py
	@echo ""
	@echo "Running advanced example..."
	python examples/advanced_usage.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 accelera/ examples/ tests/ --max-line-length=100
	black --check accelera/ examples/ tests/

format:
	black accelera/ examples/ tests/

docs:
	@echo "Documentation available in DOCUMENTATION.md"
	@echo "Opening documentation..."
	@python -c "import webbrowser; webbrowser.open('file://$(PWD)/DOCUMENTATION.md')" || echo "Please open DOCUMENTATION.md manually"

# Development targets
dev-install:
	pip install -e ".[dev]"

dev-test:
	python -m pytest tests/ -v --cov=accelera --cov-report=html

# Quick verification that everything works
verify:
	@echo "🔍 Verifying Accelera installation..."
	@python -c "import accelera as acc; print(f'✅ Accelera {acc.__version__} imported successfully')"
	@python -c "import torch; print(f'✅ PyTorch {torch.__version__} available')"
	@python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')"
	@echo "🎉 Verification complete!"

# Quick demo
demo:
	@echo "🚀 Running quick demo..."
	@python -c "\
import accelera as acc; \
import torch; \
print('Creating 1000x800 and 800x1200 matrices...'); \
engine = acc.MatrixEngine(enable_progress=False); \
A = acc.Matrix.randn((1000, 800)); \
B = acc.Matrix.randn((800, 1200)); \
C = engine.matmul(A, B); \
print(f'✅ Success! Result shape: {C.shape}'); \
print(f'Memory info: {engine.get_memory_info()[\"gpu_utilization\"]:.1f}% GPU used')"