# Configuring your compute environment

1. `sudo chmod a+w /home/jupyter`
1. `mkdir /home/jupyter/.ssh`
1. `ssh-keygen` --> select `/home/jupyter/.ssh`
1. Add key to github
1. Add `export GIT_SSH_COMMAND="ssh -i /home/jupyter/.ssh/id_rsa"` to `~/.bashrc`
1. Clone repo
1. Install poetry `curl -sSL https://install.python-poetry.org | python3 -`
1. Add poetry to bashrc `export PATH="/home/<YOUR-FIRST-NAME>/.local/bin:$PATH"`
1. Restart terminal
1. Create dir: `mkdir -p /home/jupyter/pypoetry/.cache`
1. Configure poetry
    1. `poetry config virtualenvs.in-project true`
    1. `poetry config cache-dir /home/jupyter/pypoetry/.cache`
1. `poetry install --with docs -E evaluation -E training && poetry self add poetry-git-version-plugin`
