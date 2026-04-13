import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from math import pi


def draw_triple_radar_with_labels(all_data):
    labels = ['Precision', 'Recall', 'F1-Score']
    num_vars = len(labels)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig = plt.figure(figsize=(18, 7.5))

    for i, (title, scores) in enumerate(all_data.items()):
        ax = fig.add_subplot(1, 3, i + 1, polar=True)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)


        # Extract data and close the polygon
        data_a = scores[0]
        data_b = scores[1]
        a_vals = data_a + [data_a[0]]
        b_vals = data_b + [data_b[0]]


        # Draw the chart
        ax.plot(angles, a_vals, color='#2c3e50', linewidth=2, label='Strategy A (TF-IDF)', marker='o', markersize=5)
        ax.fill(angles, a_vals, color='#2c3e50', alpha=0.1)
        ax.plot(angles, b_vals, color='#e74c3c', linewidth=2, linestyle='--', label='Strategy B (SpaCy)', marker='s',
                markersize=5)
        ax.fill(angles, b_vals, color='#e74c3c', alpha=0.1)

        for j in range(num_vars):
            angle_rad = angles[j]

            outer_r = 0.60

            # Adjust horizontal alignment based on angle
            ha_align = 'center'
            if 0.1 < angle_rad < 3.0:  # 右半边 / Right side
                ha_align = 'left'
            elif 3.2 < angle_rad < 6.0:  # 左半边 / Left side
                ha_align = 'right'

            # Label Strategy A place above the outer ring
            ax.text(angle_rad, outer_r + 0.05, f'A:{data_a[j]:.2f}',
                    color='#2c3e50', fontweight='bold',
                    ha=ha_align, va='center', fontsize=10)

            # Label Strategy B  place below the outer ring
            ax.text(angle_rad, outer_r, f'B:{data_b[j]:.2f}',
                    color='#e74c3c', fontweight='bold',
                    ha=ha_align, va='center', fontsize=10)


        # Detail beautification
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontweight='bold', fontsize=11)


        # Increase the spacing of indicator text
        ax.tick_params(axis='both', which='major', pad=15)


        plt.ylim(0, 0.75)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6], ["0.2", "0.4", "0.6"], color="grey", size=9)

        # Subplot main title
        ax.set_title(title, size=16, fontweight='bold', pad=45, y=1.1)

        # Place the legend uniformly at the bottom center
        if i == 1:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=True)


    plt.tight_layout()

    fig.subplots_adjust(top=0.78, bottom=0.18)

    plt.savefig('pico_final_labeled.png', dpi=300, bbox_inches='tight')
    plt.show()

#  Insert real data
data = {
    'Participants': [[0.42, 0.49, 0.44], [0.40, 0.67, 0.42]],
    'Interventions': [[0.34, 0.36, 0.33], [0.31, 0.38, 0.32]],
    'Outcomes': [[0.54, 0.64, 0.58], [0.42, 0.64, 0.46]]
}

draw_triple_radar_with_labels(data)