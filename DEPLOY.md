# Deploying VernaSolver to Render (free, no domain needed)

Render runs the **entire backend** (FastAPI + ChromaDB + embeddings + LLM calls)
in the cloud and gives you a free public HTTPS URL like
`https://vernasolver.onrender.com`. Your PC does nothing — you can turn it off.

---

## One-time setup

### 1. Push the project to GitHub

If you haven't already created a GitHub repo:

1. Go to https://github.com/new and make a repo (e.g. `vernasolver`). Leave it empty.
2. In the project folder, run:

   ```
   git remote add origin https://github.com/YOUR_USERNAME/vernasolver.git
   git add -A
   git commit -m "Add Render deployment config + ingested books"
   git push -u origin main
   ```

   > The books (`books/`) and vector DB (`db/`) are now committed on purpose, so
   > they ship with the deploy. Render's free tier wipes any files not in the repo
   > on every restart — committing them is what makes your books survive.

### 2. Create the Render service

1. Sign up at https://render.com with your GitHub account (free, no card).
2. Click **New +** → **Blueprint**.
3. Select your `vernasolver` repo. Render reads `render.yaml` automatically.
4. It will show one service named `vernasolver`. Click **Apply**.

### 3. Add your API key

During (or after) creation, Render asks for the env vars marked `sync: false`:

- **ANTHROPIC_API_KEY** — paste your Claude key (the same one from your local `.env`)
- **OPENAI_API_KEY** — optional fallback; leave blank if you don't use it

(You can edit these any time under the service's **Environment** tab.)

### 4. Wait for the first build

The first deploy takes **5–10 minutes** (it downloads PyTorch — ~180 MB).
When it finishes, Render shows your live URL at the top:

```
https://vernasolver.onrender.com
```

Open it. Done.

---

## Adding new books later

Because the free tier has no persistent disk, **uploading a book through the live
admin panel will not survive a restart.** To add a book permanently:

1. Ingest it locally on your PC:

   ```
   python chatbot.py ingest "books/NewBook.pdf" --subject "Subject" --title "Title" --author "Author"
   ```

2. Commit and push the updated data:

   ```
   git add books/ db/
   git commit -m "Add NewBook"
   git push
   ```

3. Render auto-redeploys with the new book in ~5 min.

---

## Good to know about the free tier

- **Sleeps after 15 min idle.** The first request after it sleeps takes ~30–50s to
  wake up (you'll see a loading spinner). After that it's fast. Fine for demos and
  college projects.
- **User accounts are per-deploy.** `db/users.db` is gitignored, so accounts created
  on the live site reset whenever Render rebuilds. If you need persistent accounts,
  upgrade to a paid plan with a disk, or move users to a hosted database — ask and
  I'll wire it up.
- **750 free hours/month** — enough to run one service continuously.

---

## If a build fails

Check the **Logs** tab in Render. The most common issue is a dependency version —
`requirements.txt` is already pinned to versions known to work on Linux, so it
should build cleanly. If torch fails, tell me the error and I'll adjust the pin.
