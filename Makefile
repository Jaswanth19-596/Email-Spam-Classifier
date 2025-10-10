format-check:
	black --check .

linting-check:
	ruff check .

security-check:
	bandit -r . -ll --exclude ./venv,./env,./.venv

format:
	black .

lint:
	ruff check . --fix

