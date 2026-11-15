# Hosting the full stack on a free Hugging Face Space

This is the fallback if Vercel is unsuitable (for example, bundle size, Python runtime issues, or a blocked gateway).
It runs the **full** stack in one Docker container: Chroma for vectors, FastEmbed for embeddings, and the showcase
FastAPI app on port 7860. The image is built from `deploy/huggingface/Dockerfile`.

## Steps

1. **Create a free Hugging Face account and a Space.** Sign up at huggingface.co, then choose *New Space*. Pick
   **Docker** as the SDK, start from the blank template, and use the free CPU hardware. Give the Space a name.

2. **Put the repo files in the Space.** A Space is a git repo. It must have the Dockerfile at its root and a
   `README.md` that starts with the YAML header. From a checkout of this repo:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-user>/<your-space>
   # Use your username and a Hugging Face access token with write scope as the password.
   git push space main
   ```
   Then, in the Space repo (not in this repo's `main`), copy the build file to the root and add the header:
   ```bash
   cp deploy/huggingface/Dockerfile Dockerfile
   # Put the YAML block from deploy/huggingface/space_readme_header.md at the top of README.md
   ```
   The COPY paths in the Dockerfile are relative to the repo root, so the build context must be the whole repo.
   `git push` only sends committed files, and `.env` is git-ignored, so your local secrets are not uploaded.

3. **Add the provider keys as Secrets.** In the Space, go to *Settings → Variables and secrets* and add one
   **secret** per key you use. The names match `.env.example` (for example `ANTHROPIC_API_KEY`, `GROQ_API_KEY`,
   `APP_API_KEY`). Secrets are passed to the running container as environment variables and are not stored in the
   repo or the image. `RETRIEVER=vector` is already set in the Dockerfile.

4. **Wait for the build.** The first build installs `requirements-full.txt` and downloads the embedding model, so it
   takes several minutes. Watch *Logs → Build*. When the status says *Running*, the app is up. If the build fails,
   check the log for a missing path (for example `showcase/` not committed yet) or a dependency error.

5. **Open the URL.** The Space serves the app at `https://<your-user>-<your-space>.hf.space`. Swagger UI is at
   `/docs`. Test one question on the RAG page before you present.

## Honest notes

- **Free CPU Spaces sleep when idle.** After a period with no traffic the Space stops, and the first request after
  that waits while it restarts. Open the URL a few minutes before a live demo.
- **The disk is not persistent.** Anything written at runtime (chat threads, traces, uploaded files) is lost on
  restart or rebuild. The showcase keeps agent threads in memory, so they also reset on restart.
- **Keys are billed to you.** Every LLM call uses the keys you added, so set provider budgets and rotate keys if you
  share the URL.
- Free hardware limits (CPU, RAM) are set by Hugging Face and can change. Re-check them on the Spaces pricing page.

## Self-host alternative (documented only, not tested)

As a self-host alternative, an Oracle Cloud Always-Free VM can run the same Dockerfile under Coolify or Dokku, which add a
deploy UI and HTTPS. Oracle's free tier has its own capacity and signup rules, so check them before you rely on it.
