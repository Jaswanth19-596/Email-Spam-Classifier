format-check:
	black --check .

linting-check:
	ruff check .

security-check:
	bandit -r . -ll --exclude ./venv,./env,./.venv

format-fix:
	black .

lint-fix:
	ruff check . --fix

