import matplotlib.pyplot as plt
import numpy as np

# Define simulator types and their counts
simulators = ['Analysis', 'Custom', 'NetSquid', 'Matlab', 'Sequence']
counts = [12, 13, 5, 1, 2]  # from the provided data (33 total)

# Calculate percentages
total = sum(counts)
percentages = [100 * count / total for count in counts]
# Group percentages: Custom + Analysis and Public Simulators (NetSquid, Matlab, Sequence)
custom_analysis_pct = percentages[0] + percentages[1]
public_pct = sum(percentages[2:])

# Colors: Use one palette for in-house models and one for public simulators
colors = ['#4C72B0', '#4C72B0', '#55A868', '#55A868', '#55A868']  # blue for custom/analysis, green for public

# Prepare stacked bar values
bottom = 0
x = [0]  # single bar
plt.figure(figsize=(6, 8))
for pct, sim, color in zip(percentages, simulators, colors):
    plt.bar(x, pct, bottom=bottom, color=color, edgecolor='white', width=0.5, label=sim)
    # Annotate each segment with its percentage and simulator type
    plt.text(x[0], bottom + pct/2, f"{sim}\n({pct:.1f}%)", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    bottom += pct

# Draw horizontal line to separate the two groups for clarity
# (Custom + Analysis vs Public Simulators)
plt.axhline(custom_analysis_pct, color='black', linestyle='--', linewidth=1)
plt.text(0.6, custom_analysis_pct - 2, f'Custom & Analysis: {custom_analysis_pct:.1f}%', color='black', fontsize=10)
plt.text(0.6, custom_analysis_pct + 2, f'Public Simulators: {public_pct:.1f}%', color='black', fontsize=10)

plt.xticks([])  # Hide x-axis ticks for a single bar
plt.ylabel("Percentage of Studies (%)")
plt.title("Distribution of Simulator Approaches in Quantum Network Research\n(Custom/Analysis vs Generally Available Simulators)")
plt.ylim(0, 110)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
