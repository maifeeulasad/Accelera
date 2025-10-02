# Makefile for Accelera development

.PHONY: help install test benchmark examples clean lint docs

help:
	@echo "Accelera Development Commands:"
	@echo "  install     - Install package in development mode"
	@echo "  test        - Run unit tests"
	@echo "  benchmark   - Run performance benchmarks"
	@echo "  examples    - Run example scripts"
	@echo "  clean       - Clean up build artifacts"
	@echo "  lint        - Run code linting"
	@echo "  docs        - Build documentation"

install:
	pip install -r requirements.txt
	pip install -e .

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