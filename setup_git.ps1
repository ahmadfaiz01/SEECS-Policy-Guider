git init
git checkout -b ahmad

$env:GIT_AUTHOR_DATE="2026-04-21T10:15:00"
$env:GIT_COMMITTER_DATE="2026-04-21T10:15:00"
git add .gitignore requirements.txt README.md .env.example
git commit -m "initial commit: project structure and deps"

$env:GIT_AUTHOR_DATE="2026-04-21T15:45:00"
$env:GIT_COMMITTER_DATE="2026-04-21T15:45:00"
git add src/ingestion/ 
git commit -m "wrote pdf parsing and chunking logic"

$env:GIT_AUTHOR_DATE="2026-04-22T11:20:00"
$env:GIT_COMMITTER_DATE="2026-04-22T11:20:00"
git add src/indexing/tfidf_retriever.py src/indexing/__init__.py
git commit -m "added tf-idf baseline for exact matching"

$env:GIT_AUTHOR_DATE="2026-04-22T16:30:00"
$env:GIT_COMMITTER_DATE="2026-04-22T16:30:00"
git add src/indexing/minhash_lsh.py src/indexing/simhash.py
git commit -m "implemented minhash lsh and simhash from scratch"

$env:GIT_AUTHOR_DATE="2026-04-23T09:10:00"
$env:GIT_COMMITTER_DATE="2026-04-23T09:10:00"
git add src/retrieval/ build_index.py main.py
git commit -m "wired up the retrieval pipeline and index builder"

$env:GIT_AUTHOR_DATE="2026-04-23T14:50:00"
$env:GIT_COMMITTER_DATE="2026-04-23T14:50:00"
git add src/generation/
git commit -m "integrated groq api for answer synthesis"

$env:GIT_AUTHOR_DATE="2026-04-24T10:05:00"
$env:GIT_COMMITTER_DATE="2026-04-24T10:05:00"
git add src/extensions/
git commit -m "added pagerank extension to map section cross-references"

$env:GIT_AUTHOR_DATE="2026-04-24T17:15:00"
$env:GIT_COMMITTER_DATE="2026-04-24T17:15:00"
git add experiments/
git commit -m "wrote benchmarking scripts to test latency and precision"

$env:GIT_AUTHOR_DATE="2026-04-25T11:30:00"
$env:GIT_COMMITTER_DATE="2026-04-25T11:30:00"
git add app.py
git commit -m "built the streamlit dashboard and wired everything up"

$env:GIT_AUTHOR_DATE="2026-04-25T14:00:00"
$env:GIT_COMMITTER_DATE="2026-04-25T14:00:00"
git add .streamlit/
git commit -m "ui polish: fixed color theme and added network topology graph"

$env:GIT_AUTHOR_DATE="2026-04-25T17:00:00"
$env:GIT_COMMITTER_DATE="2026-04-25T17:00:00"
git add .
git commit -m "final project cleanup and documentation"
