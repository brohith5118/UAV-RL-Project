# Test baseline script
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Create artifacts directory if not exists
art_dir = r"C:\Users\emand\.gemini\antigravity-ide\brain\dab6e8c9-bc2b-4e5d-afc9-6fee29422b1b"
os.makedirs(art_dir, exist_ok=True)

fig_counter = 0
def save_instead_of_show():
    global fig_counter
    fig_counter += 1
    path = os.path.join(art_dir, f"baseline_fig_{fig_counter}.png")
    plt.savefig(path, dpi=150)
    print(f"Saved figure to {path}")
    plt.close()

plt.show = save_instead_of_show

# Now run the main module
import main
if __name__ == '__main__':
    main.main()
