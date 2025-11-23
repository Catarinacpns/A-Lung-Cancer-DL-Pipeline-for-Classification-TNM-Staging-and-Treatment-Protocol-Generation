# Function to sample data with focus on maximizing number of images while keeping all categories
def relaxed_stratified_sample(df, max_images, max_memory, stage_columns, seed=1):
    # Set a seed for reproducibility
    np.random.seed(seed)

    # Drop rows with missing stage values (if any)
    df = df.dropna(subset=stage_columns)

    # Group by stage columns to ensure each category is present at least once
    grouped = df.groupby(stage_columns)

    # Sample each group to ensure at least one sample per group, if possible
    sampled_df = pd.DataFrame()
    total_images_sampled = 0
    total_memory_used = 0
    
    # First ensure all groups are represented by at least one image
    for group_name, group in grouped:
        if not group.empty:
            sample = group.sample(n=min(1, len(group)), random_state=seed)  # Ensure at least one row per group if possible
            sampled_df = pd.concat([sampled_df, sample])
            total_images_sampled += sample['Number of Images'].sum()
            total_memory_used += sample['File Size'].sum()

    # Sample the rest of the data prioritizing the image and memory limits for balancing
    remaining_images = max_images - total_images_sampled
    remaining_memory = max_memory - total_memory_used

    if remaining_images > 0 and remaining_memory > 0:
        # Sort by file size for efficient memory usage
        remaining_df = df[~df.index.isin(sampled_df.index)].sort_values(by='File Size')
        remaining_df = remaining_df[(remaining_df['File Size'].cumsum() <= remaining_memory) & 
                                    (remaining_df['Number of Images'].cumsum() <= remaining_images)]
        
        sampled_df = pd.concat([sampled_df, remaining_df])
        total_images_sampled += remaining_df['Number of Images'].sum()
        total_memory_used += remaining_df['File Size'].sum()

    return sampled_df, total_memory_used


# Function to add more patients if space remains
def add_more_patients(df, sampled_df, remaining_images, remaining_memory, seed=19):
    if remaining_images > 0 and remaining_memory > 0:
        remaining_df = df[~df.index.isin(sampled_df.index)].sort_values(by='File Size')
        additional_sample = remaining_df[
            (remaining_df['File Size'].cumsum() <= remaining_memory) & 
            (remaining_df['Number of Images'].cumsum() <= remaining_images)
        ]
        sampled_df = pd.concat([sampled_df, additional_sample])
    return sampled_df


# Modify the get_target_sample function to keep all patients in B and E, and sample A and G
def get_target_sample(dfs, max_images_per_df, remaining_memory, stage_columns, seed=1):
    sampled_dfs = {'E': dfs['E'], 'B': dfs['B']}  # Include all data from E and B
    remaining_memory -= dfs['E']['File Size'].sum()  # Update remaining memory after including E dataset
    remaining_memory -= dfs['B']['File Size'].sum()  # Update remaining memory after including B dataset
    
    # Now we sample for A and G
    for key in ['A', 'G']:
        df_meta = dfs[key]
        max_images = min(df_meta['Number of Images'].sum(), max_images_per_df)  # Max of 28,000 images or less if fewer available
        max_memory = remaining_memory / len(['A', 'G'])  # Distribute remaining memory dynamically between A and G

        # Use relaxed stratified sampling to maximize images while keeping some stage diversity
        valid_sample, memory_used = relaxed_stratified_sample(df_meta, max_images, max_memory, stage_columns, seed=seed)

        # Store sampled data and adjust remaining memory
        sampled_dfs[key] = valid_sample
        remaining_memory -= memory_used
        
        # After getting the balanced sample, try to add more patients if space remains
        remaining_images = max_images_per_df - sampled_dfs[key]['Number of Images'].sum()
        sampled_dfs[key] = add_more_patients(df_meta, sampled_dfs[key], remaining_images, remaining_memory, seed=seed)

    return sampled_dfs, remaining_memory



# Adjust sample sizes if necessary
def adjust_sample_size(sampled_dfs, min_memory_limit_mb, max_memory_limit_mb):
    total_memory_used = sum(df['File Size'].sum() for key, df in sampled_dfs.items())
    
    while total_memory_used < min_memory_limit_mb:
        for key in ['A', 'G']:  # Only adjust for A and G
            if total_memory_used >= max_memory_limit_mb:
                break
            df_meta = dfs[key]
            additional_sample = df_meta[~df_meta.index.isin(sampled_dfs[key].index)]
            if additional_sample.empty:
                continue
            additional_sample = additional_sample.sort_values(by='File Size').iloc[:1]
            sampled_dfs[key] = pd.concat([sampled_dfs[key], additional_sample])
            total_memory_used += additional_sample['File Size'].sum()
        
    while total_memory_used > max_memory_limit_mb:
        for key in ['A', 'G']:  # Only adjust for A and G
            if total_memory_used <= min_memory_limit_mb:
                break
            df_meta = sampled_dfs[key]
            if len(df_meta) > 1:
                # Use iloc instead of index to drop the largest file size row
                largest_file_index = df_meta['File Size'].idxmax()
                largest_file_pos = df_meta['File Size'].sort_values(ascending=False).index[0]
                total_memory_used -= df_meta.loc[largest_file_pos, 'File Size']
                df_meta = df_meta.drop(largest_file_pos)
                sampled_dfs[key] = df_meta
    
    return sampled_dfs, total_memory_used



# Function to calculate the distribution of stage columns
def calculate_stage_distribution(df, stage_columns):
    # Group by the stage columns and count the occurrences of each combination
    distribution = df.groupby(stage_columns).size().reset_index(name='count')
    
    # Calculate the proportions (percentage) for each combination
    distribution['proportion'] = distribution['count'] / distribution['count'].sum()
    
    return distribution



# Function to compare distributions between the original and sampled datasets
def compare_distributions(original_df, sampled_df, stage_columns):
    # Calculate the distribution for the original and sampled data
    original_distribution = calculate_stage_distribution(original_df, stage_columns)
    sampled_distribution = calculate_stage_distribution(sampled_df, stage_columns)

    # Merge the two distributions on the stage columns
    comparison = pd.merge(original_distribution, sampled_distribution, on=stage_columns, suffixes=('_original', '_sampled'), how='outer').fillna(0)

    # Compute the absolute difference in proportions
    comparison['proportion_diff'] = abs(comparison['proportion_original'] - comparison['proportion_sampled'])
    
    # Compute the relative difference in proportions (percentage difference)
    comparison['relative_diff_percentage'] = (comparison['proportion_diff'] / comparison['proportion_original']) * 100

    return comparison



# Function to compare the balance of all datasets against the original
def check_balance(dfs, final_samples, stage_columns):
    for key in dfs.keys():
        print(f"\n--- Balance Comparison for Dataset {key} ---")
        
        # Compare the original dataset with the sampled one
        comparison = compare_distributions(dfs[key], final_samples[key], stage_columns)
        
        # Display the comparison
        display(comparison[['N-Stage', 'Ｍ-Stage', 'T-Stage', 'Histopathological grading', 'proportion_original', 'proportion_sampled', 'relative_diff_percentage']])