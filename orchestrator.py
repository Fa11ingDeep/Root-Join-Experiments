import subprocess

# List of commands to execute sequentially
jobs = [
    #"python3 root_join.py --dataset gooaq",
    "python3 root_join.py --dataset ccnews",
    "python3 root_join.py --dataset yahooaq"
]

for job in jobs:
    print(f"\n=== Executing: {job} ===")
    result = subprocess.run(job, shell=True)

    if result.returncode != 0:
        print(f"Error while executing: {job}")
        break  # Stop if any job fails
    else:
        print(f"Finished successfully: {job}\n")