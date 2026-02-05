#!/usr/bin/env python3
import os
import subprocess
import sys
import time


def run_git(args, cwd, check=True):
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def git_output(args, cwd):
    return run_git(args, cwd=cwd).stdout.strip()


def has_changes(cwd):
    status = git_output(["status", "--porcelain"], cwd)
    return bool(status)


def has_staged_changes(cwd):
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=cwd,
    )
    return result.returncode != 0


def ensure_repo_root():
    try:
        root = git_output(["rev-parse", "--show-toplevel"], os.getcwd())
        if not root:
            raise RuntimeError("Empty repo root")
        return root
    except Exception as exc:
        print(f"[auto-commit] Not a git repo: {exc}")
        sys.exit(1)


def current_branch(cwd):
    branch = git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if branch == "HEAD":
        return ""
    return branch


def has_remote(cwd, name):
    try:
        url = git_output(["remote", "get-url", name], cwd)
        return bool(url)
    except subprocess.CalledProcessError:
        return False


def has_head(cwd):
    try:
        run_git(["rev-parse", "--verify", "HEAD"], cwd=cwd)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    interval = float(os.environ.get("AUTO_COMMIT_INTERVAL", "2"))
    message = os.environ.get("AUTO_COMMIT_MESSAGE", "auto: update")
    remote = os.environ.get("AUTO_COMMIT_REMOTE", "origin")

    repo_root = ensure_repo_root()
    print(f"[auto-commit] Watching {repo_root}")
    print(f"[auto-commit] Interval: {interval}s, remote: {remote}")

    while True:
        try:
            if not has_changes(repo_root):
                time.sleep(interval)
                continue

            if not has_head(repo_root):
                print("[auto-commit] No commits yet; waiting for initial commit.")
                time.sleep(interval)
                continue

            run_git(["add", "-A"], cwd=repo_root)
            if not has_staged_changes(repo_root):
                time.sleep(interval)
                continue

            run_git(["commit", "-m", message], cwd=repo_root)
            if has_remote(repo_root, remote):
                branch = current_branch(repo_root)
                if branch:
                    run_git(["push", remote, branch], cwd=repo_root)
                else:
                    print("[auto-commit] Detached HEAD; skip push.")
            else:
                print(f"[auto-commit] Remote '{remote}' not found; skip push.")
        except subprocess.CalledProcessError as exc:
            print(f"[auto-commit] git error: {exc.stderr.strip()}")
        except Exception as exc:
            print(f"[auto-commit] error: {exc}")

        time.sleep(interval)


if __name__ == "__main__":
    main()
