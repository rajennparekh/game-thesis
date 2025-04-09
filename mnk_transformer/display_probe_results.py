def launch_probe_accuracy_viewer(csv_path='../one_head_probe.csv'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    # Load and prepare the data
    df = pd.read_csv(csv_path)
    task_map = {0: 4, 1: 3, 2: 1, 3: 2}
    df['Task'] = df['Task'].map(task_map)

    melted_df = df.melt(
        id_vars=['Layer', 'Checkpoint', 'Task'],
        value_vars=[
            'Linear SA', 'Small MLP SA', 'Large MLP SA'
        ],
        var_name='Model_Metric',
        value_name='Accuracy'
    )
    melted_df['Model'] = pd.Categorical(
        melted_df['Model_Metric'].str.extract(r'(Linear|Small MLP|Large MLP)')[0],
        categories=['Linear', 'Small MLP', 'Large MLP'],
        ordered=True
    )

    all_vars = ['Layer', 'Checkpoint', 'Task']
    x_selector = widgets.Dropdown(options=all_vars, description='X-axis:')

    output = widgets.Output()

    def update_plot(*args):
        output.clear_output()
        x_var = x_selector.value
        remaining_vars = [v for v in ['Layer', 'Checkpoint', 'Task', 'Model'] if v != x_var]
        fixed1, fixed2, group_var = remaining_vars

        unique1 = sorted(melted_df[fixed1].unique())
        unique2 = sorted(melted_df[fixed2].unique())
        num_rows = len(unique1)
        num_cols = len(unique2)

        with output:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4.8), squeeze=False)

            for i, row_val in enumerate(unique1):
                for j, col_val in enumerate(unique2):
                    ax = axes[i][j]
                    subset = melted_df[(melted_df[fixed1] == row_val) & (melted_df[fixed2] == col_val)]
                    legend_handles = []
                    legend_labels = []

                    if x_var == 'Task':
                        models = melted_df['Model'].cat.categories
                        bar_width = 0.25
                        offset_range = [-bar_width, 0, bar_width]
                        for idx, model in enumerate(models):
                            model_subset = subset[subset['Model'] == model]
                            bars = model_subset.groupby('Task')['Accuracy'].mean().sort_index()
                            positions = bars.index + offset_range[idx]
                            bars_plot = ax.bar(positions, bars.values, width=bar_width, label=str(model))
                            legend_handles.append(bars_plot[0])
                            legend_labels.append(str(model))
                        ax.set_xticks(sorted(melted_df['Task'].unique()))
                        ax.set_xticklabels([f'Task {t}' for t in sorted(melted_df['Task'].unique())])
                    else:
                        for grp_val in melted_df[group_var].cat.categories if group_var == "Model" else sorted(subset[group_var].unique()):
                            line = subset[subset[group_var] == grp_val]
                            grouped = line.groupby(x_var)['Accuracy'].mean().sort_index()
                            line_plot, = ax.plot(grouped.index, grouped.values, marker='o', label=str(grp_val))
                            legend_handles.append(line_plot)
                            legend_labels.append(str(grp_val))

                        # Force integer x-ticks
                        unique_x = sorted(subset[x_var].unique())
                        ax.set_xticks(unique_x)

                    # Y-axis scaling
                    if 'Task' in [fixed1, fixed2]:
                        task_val = row_val if fixed1 == 'Task' else col_val
                        if task_val == 2:
                            ax.set_ylim(0.9, 1)
                        else:
                            ax.set_ylim(0.5, 1)
                    elif 'Model' in [fixed1, fixed2]:
                        ax.set_ylim(0.5, 1)

                    ax.set_title(f'{fixed1} = {row_val}, {fixed2} = {col_val}')
                    ax.set_xlabel(x_var)
                    ax.set_ylabel('Space Accuracy')
                    ax.grid(True, axis='y')
                    ax.yaxis.get_major_locator().set_params(integer=True)

                    # Legend below the subplot with spacing
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0 + 0.13, box.width, box.height * 0.8])
                    ax.legend(handles=legend_handles, labels=legend_labels, loc='upper center',
                              bbox_to_anchor=(0.5, -0.32), ncol=len(legend_labels), fontsize=9, title=group_var)

            # Top column labels
            for j, col_val in enumerate(unique2):
                fig.text(
                    x=(j + 0.5) / num_cols,
                    y=1.02,
                    s=f'{fixed2} = {col_val}',
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    fontweight='bold'
                )

            # Left row labels
            for i, row_val in enumerate(unique1):
                fig.text(
                    x=-0.01,
                    y=(num_rows - i - 0.5) / num_rows,
                    s=f'{fixed1} = {row_val}',
                    ha='right',
                    va='center',
                    fontsize=12,
                    fontweight='bold',
                    rotation=90
                )

            plt.tight_layout()
            plt.show()

    x_selector.observe(update_plot, names='value')
    display(widgets.VBox([x_selector, output]))
    update_plot()
