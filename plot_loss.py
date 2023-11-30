
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def handle_missing_data(df, column_name):
    # Check for NaN values and fill with the previous value
    if df[column_name].isnull().any():
        print(f"NaN values found in {column_name}. Filling with previous values.")
        df[column_name].fillna(method='ffill', inplace=True)

def analyze_zero_values(csv_file, columns):
    df = pd.read_csv(csv_file)

    for column in columns:
        # Count the total number of zero values in the column
        total_zeros = (df[column] == 0).sum()
        print(f"Total zero values in {column}: {total_zeros}")

        # Find the maximum number of consecutive zero values
        max_consecutive_zeros = ((df[column] == 0).astype(int).groupby(df[column].ne(0).astype(int).cumsum()).sum()).max()
        print(f"Maximum consecutive zero values in {column}: {max_consecutive_zeros}\n")
              
def analyze_nan_values(csv_file, columns):
    df = pd.read_csv(csv_file)

    for column in columns:
        # Count the total number of NaN values in the column
        total_nan = df[column].isnull().sum()
        print(f"Total NaN values in {column}: {total_nan}")

        # Find the maximum number of consecutive NaN values
        max_consecutive_nan = (df[column].isnull().astype(int).groupby(df[column].notnull().astype(int).cumsum()).sum()).max()
        print(f"Maximum consecutive NaN values in {column}: {max_consecutive_nan}\n")

def plot_losses(df, outliers_threshold = 20):
    
    # Handle missing data for 'SumLoss' and 'contrastive_loss'
    handle_missing_data(df, 'SumLoss')
    handle_missing_data(df, 'contrastive_loss')
    
    # Remove outliers from 'contrastive_loss'
    df = df[df['contrastive_loss'] < outliers_threshold]
    
    # Calculate mean of 'contrastive_loss' after removing outliers
    mean_contrastive_loss = df['contrastive_loss'].mean()
    
    # Calculate mean of 'SumLoss' for iterations after 1000
    mean_sum_loss_after_1000 = df[df['Iter'] > 1000]['SumLoss'].mean()

    # Filter rows for every ten iterations
    filtered_df = df[df['Iter'] % 5 == 0]

    # Plotting
    plt.figure(figsize=(10, 5))

    # Plot SumLoss
    plt.plot(filtered_df['Iter'], filtered_df['SumLoss'], label='Sum Loss')
    # Add a horizontal line for the mean of SumLoss after 1000 iterations
    plt.axhline(y=mean_sum_loss_after_1000, color='g', linestyle='--', label='Mean Sum Loss after 1000 Iterations')

    # Plot Contrastive Loss if it exists in the dataframe
    if 'contrastive_loss' in filtered_df.columns:
        plt.plot(filtered_df['Iter'], filtered_df['contrastive_loss'], label='Contrastive Loss')
        # Add a horizontal line for the mean of contrastive_loss
        plt.axhline(y=mean_contrastive_loss, color='r', linestyle='--', label='Mean Contrastive Loss')


    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss per Iteration')
    plt.legend()
    plt.grid(True)
    # plt.show()
    
    # Set y-axis ticks with an interval of 5
    y_max = max(filtered_df['SumLoss'].max(), filtered_df['contrastive_loss'].max())
    y_ticks = (np.arange(0, y_max, 5))
    y_ticks = np.append(y_ticks, [mean_sum_loss_after_1000, mean_contrastive_loss])  # I
    
    plt.yticks(y_ticks)
    
    # Save plot to a file
    plt.savefig("loss_plot.png")
    plt.close()  # Close the figure to free memory


def main(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Call the plot function
    plot_losses(df)

# Example usage
if __name__ == "__main__":
    csv_file_name = "training_csv.csv"  # Replace with your CSV file name
    columns_to_check = ['SumLoss', 'contrastive_loss']  # Replace with your column names
    analyze_nan_values(csv_file_name, columns_to_check)
    main(csv_file_name)
