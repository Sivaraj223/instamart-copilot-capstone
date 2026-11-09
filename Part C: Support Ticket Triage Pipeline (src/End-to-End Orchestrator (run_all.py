import subprocess
import sys

def run_cmd(cmd):
    print(f"\n--- Running: {cmd} ---")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"Failed at: {cmd}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    run_cmd("python generate_data.py")
    run_cmd("python src/part_a_pipeline.py")
    run_cmd("python src/part_a_validate.py")
    run_cmd("python src/part_b_pipeline.py")
    run_cmd("python src/part_c_pipeline.py")
    run_cmd("python src/part_d_pipeline.py")
    print("\n Instamart Copilot execution complete.")
