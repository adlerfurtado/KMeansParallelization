import random
import sys

def generate_points(N, min_val=-1000.0, max_val=1000.0):
    filename = f"./data/random_{N}.txt"
    with open(filename, "w") as f:
        for _ in range(N):
            x = random.uniform(min_val, max_val)
            y = random.uniform(min_val, max_val)
            f.write(f"{x} {y}\n")

    print(f"Arquivo '{filename}' gerado com {N} pontos.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python generate_points.py <N> [<min> <max>]")
        sys.exit(1)

    N = int(sys.argv[1])
    if len(sys.argv) == 4:
        generate_points(N, float(sys.argv[2]), float(sys.argv[3]))
    else:
        generate_points(N)
