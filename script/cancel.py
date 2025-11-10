# get all slurm jobs through squeue --me and then scancel each of them
import subprocess


def cancel_all_slurm_jobs():
    try:
        # Get all SLURM jobs for the current user
        result = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%A"], capture_output=True, text=True, check=True
        )
        job_ids = result.stdout.strip().split("\n")

        if not job_ids or job_ids == [""]:
            print("No SLURM jobs found.")
            return

        # Cancel each job
        for job_id in job_ids:
            subprocess.run(["scancel", job_id], check=True)
            print(f"Cancelled SLURM job {job_id}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    cancel_all_slurm_jobs()
    print("All SLURM jobs cancelled.")
else:
    print("This script is intended to be run directly, not imported.")
