import os

from app import create_app

app = create_app()

if os.environ.get("WARM_MODEL", "1") == "1":
    # Load the face model once at startup so the first student does not wait.
    try:
        import face_engine

        face_engine.get_engine()
        print("Face model loaded.", flush=True)
    except Exception as exc:  # the site still starts; scans will report the error
        print("WARNING: face model failed to load:", exc, flush=True)
