"""Run on your own computer: python run_local.py  ->  http://localhost:5000"""
import os

os.environ.setdefault("INSECURE_COOKIES", "1")  # plain http on your own machine
from app import create_app  # noqa: E402

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=False)
