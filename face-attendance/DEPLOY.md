# Deploy guide (Hugging Face Spaces, free)

## 1. Free database (so data survives restarts)
Free Spaces lose local files when they restart, so keep data in a free Postgres.
1. Create a free project at https://neon.tech and copy the connection string
   (looks like `postgresql://user:pass@host/db?sslmode=require`).

## 2. Create the Space
1. https://huggingface.co/new-space -> SDK **Docker** -> blank -> Public or Private.
2. Upload every file in this folder (Files tab -> Add file -> Upload). Keep `README.md` as is
   (its first lines tell Hugging Face to run the Dockerfile).
3. Space -> **Settings -> Variables and secrets -> New secret**, add:
   | Name | Value |
   |---|---|
   | `SECRET_KEY` | any long random text |
   | `ADMIN_PASSWORD` | the password you want for the admin account |
   | `DATABASE_URL` | the Neon connection string |
   Optional variables: `ADMIN_USERNAME` (default `admin`), `MATCH_THRESHOLD` (default 0.45),
   `YAW_DELTA` (default 0.12), `LIVENESS_DEBUG=1` (shows head-turn numbers on failed scans while you tune).
4. The first build takes several minutes (it compiles InsightFace and downloads the model).

## 3. Use it
- Open the **direct** URL `https://<your-name>-<space-name>.hf.space` (not the huggingface.co page).
  The embedded page blocks cookies/camera on many phones.
- Log in as admin -> add the teacher and students (or paste a CSV) -> hand out the shown passwords
  -> **Register face** for each student (consent box, 6 guided photos).
- Teacher: Sessions -> Start now. Students: log in -> My attendance -> Scan my face.

## 4. Tune on real faces (important, not tested yet)
Test with 3-4 real students on laptop and phone, in classroom lighting:
- A genuine student should pass in one or two tries.
- A photo of the student on another phone should fail ("Liveness check failed").
- If real students fail the head-turn, lower `YAW_DELTA` (try 0.08). If photos pass, raise it.
- If matching is too strict/loose, change `MATCH_THRESHOLD` (0.35 looser ... 0.55 stricter).

## Run on your own computer
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt     # insightface needs "Microsoft C++ Build Tools" on Windows
python run_local.py                 # http://localhost:5000
python -m unittest discover -s tests   # app tests (use a fake face engine)
```
First run prints the admin password if `ADMIN_PASSWORD` is not set.
Phones need HTTPS for the camera, so test phones on the deployed Space, not on your LAN.

## Known limits
- Head-turn liveness stops photos and most screen-held photos, but a recorded video of the
  student turning their head could fool it, and so could a 3D mask. Teachers can always correct
  marks manually and the report shows how each mark was made (face/manual).
- Free Spaces sleep after a period of no use; the first visit then takes about a minute.
