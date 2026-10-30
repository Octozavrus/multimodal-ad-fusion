# CI/CD with Docker and Jenkins

The repository has two CI systems that run the same checks:

* **GitHub Actions** (`.github/workflows/ci.yml`) on every push to `main` and every pull request;
* **Jenkins** (`Jenkinsfile`) on a self-hosted server defined in `ci/`, which also builds and
  deploys the Docker image.

## Pipeline

| Stage | Branches | What it does |
|---|---|---|
| Prepare | all | short commit hash, package version, Docker version |
| Data guard | all | `scripts/check_no_data.py`: no imaging files, checkpoints or real ADNI subject IDs |
| Build test image | all | `docker build --target test` (package with dev extras and tests) |
| Lint | all | `ruff check` and `ruff format --check` inside the test image |
| Unit tests | all | `pytest` inside the test image, coverage gate 85 %; JUnit and Cobertura reports are copied out with `docker cp` and published in Jenkins |
| Build image | all | `docker build --target runtime`, tagged with the commit hash and `latest` |
| Smoke test image | all | `adfusion --version` and `info`, then inside the image: synthetic scans → preprocessing → QC gate, and a synthetic cohort → nested cross-validation (smoke protocol) → figures |
| Push to Docker Hub | `main`, if `PUSH_IMAGE=true` | tags `<sha>`, `<version>` and `latest` |
| Deploy to staging | `main` | `deploy/compose.staging.yml`: the image runs the whole pipeline on synthetic data and serves the cache, the results tables and the figures on <http://localhost:8081>; the stage waits for the container health check |
| Deploy to Kubernetes | `main`, parameter `DEPLOY_K8S` | `k8s/job.yaml` as a Kubernetes Job; waits for completion and prints the log |

Tests run inside a container built from the same Dockerfile as the released image, so the
Jenkins host needs only Docker, not Python, PyTorch or SimpleITK. The first build downloads
CPU-only PyTorch (about 200 MB) into a cached image layer; later builds reuse it.

## Start the Jenkins server (Windows with Docker Desktop)

1. Install Docker Desktop (WSL 2 backend) and start it.
2. In PowerShell:

   ```powershell
   git clone https://github.com/Octozavrus/multimodal-ad-fusion.git
   cd multimodal-ad-fusion\ci
   copy .env.example .env
   notepad .env            # set JENKINS_ADMIN_PASSWORD
   docker compose up -d --build
   ```

3. Open <http://localhost:8080> and log in with the user from `.env`.
   The job **adfusion** already exists (Configuration as Code, `ci/jenkins/casc.yaml`).
   It scans the repository every 2 minutes and builds each branch whose head commit changed,
   so every pushed commit is built. Click **Scan Multibranch Pipeline Now** for the first run.

macOS and Linux are the same, with `cp` instead of `copy`.

Stop the server with `docker compose down`; the Jenkins home is kept in the `jenkins_home`
volume (`docker compose down -v` removes it).

## Optional: push to Docker Hub

Create an access token on Docker Hub, put `DOCKERHUB_USER`, `DOCKERHUB_TOKEN` and
`PUSH_IMAGE=true` in `ci/.env`, and recreate the server with `docker compose up -d`.
The next build of `main` pushes the image.

## Optional: Kubernetes

With Docker Desktop, enable **Settings → Kubernetes**. Images built by Jenkins are then
visible to the cluster. In Jenkins add a *Secret file* credential with ID `kubeconfig`
(your `%USERPROFILE%\.kube\config`) and run the `main` branch with **Build with
Parameters → DEPLOY_K8S**. With Minikube, run `minikube image load adfusion:latest`
before the deployment.

## Notes

* Builds are triggered by polling because a server on `localhost` cannot receive GitHub
  webhooks. A server with a public URL would use a webhook instead.
* The Jenkins container runs as root so that it can use the host's Docker socket. This is
  acceptable for a local single-user server, not for a shared one.
* No secret is stored in the repository: passwords and tokens are read from `ci/.env`,
  which is ignored by git.
