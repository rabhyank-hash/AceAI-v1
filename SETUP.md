# Setup (WSL)

1. **Project folder.** Unzip this kit in WSL, e.g. to `~/ace-ai` (keep it on the Linux filesystem,
   not under `/mnt/c`, for speed).
2. **Data.** From Drive → "Documents for Ruchi (Training Plan Generation)" → `sail_course_LOs`, download the six
   `*_learning_objectives_20260916.csv` files into `~/ace-ai/data/raw/`. `PPP_syllabus_broad_LOs.csv`
   is already there.
3. **Groq key.** Create a free API key at console.groq.com (no card needed). You'll paste it into
   `~/ace-ai/.env` as `GROQ_API_KEY=...` once Claude Code creates `.env.example` in Step 1.
   Note your model's rate limits from the console's Limits page; Claude Code will ask for them.
4. **Python.** Python 3.11+ with venv: `sudo apt install python3 python3-venv python3-pip`.
5. **Claude Code.** Install per https://docs.claude.com/en/docs/claude-code/setup, then run `claude`
   inside `~/ace-ai`.
6. **Kick off.** Paste the prompt from `KICKOFF.md`. It stops after each step for your review.
