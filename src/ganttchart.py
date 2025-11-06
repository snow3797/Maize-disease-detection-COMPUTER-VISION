# Creating a flipped Gantt chart: Weeks 1-4 at the top, Weeks 21-24 at the bottom.
# This will display the chart and save it to /mnt/data/gantt_flipped.png so you can download it.
import pandas as pd
import matplotlib.pyplot as plt

# Define tasks (flipped order: earliest at the top, latest at the bottom)
tasks = [
    ("Weeks 1–4: Literature review, problem definition, and requirements gathering", 1, 4),
    ("Weeks 5–8: System design and architecture development", 5, 4),
    ("Weeks 9–14: Implementation of data collection and processing modules", 9, 6),
    ("Weeks 15–20: Dashboard development and visualization integration", 15, 6),
    ("Weeks 21–24: Pilot testing and user feedback collection", 21, 4),
]

df = pd.DataFrame(tasks, columns=["Task", "StartWeek", "Duration"])
df["EndWeek"] = df["StartWeek"] + df["Duration"]

# Prepare the plot
fig, ax = plt.subplots(figsize=(10, 4.5))
y_positions = range(len(df))

# Plot horizontal bars (matplotlib will use default color cycle; we don't set colors explicitly)
ax.barh(y_positions, df["Duration"], left=df["StartWeek"], height=0.6)

# Set y labels in the same order so top = Weeks 1-4
ax.set_yticks(y_positions)
ax.set_yticklabels(df["Task"])

# Invert y-axis so the first task appears at the top
ax.invert_yaxis()

# Axis labels and title
ax.set_xlabel("Weeks")
ax.set_xlim(0, 28)
ax.set_xticks(range(0, 29, 2))
ax.set_title("Project Gantt Chart (flipped: Weeks 1–4 at top, Weeks 21–24 at bottom)")

# Add grid for readability
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)

# Annotate each bar with start and end week
for i, row in df.iterrows():
    ax.text(row["StartWeek"] + 0.2, i, f"{row['StartWeek']}–{row['EndWeek']-1}", va="center", fontsize=9)

plt.tight_layout()
outfile = "/mnt/data/gantt_flipped.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
plt.show()

# Display the dataframe to the user in a friendly table
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Gantt task table (flipped order)", df)

# Provide the path to the saved image for download
outfile