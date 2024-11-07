import os
import pandas as pd
import matplotlib

# Set non-GUI backend
matplotlib.use('Agg')  # Ensure this is the first Matplotlib line
import matplotlib.pyplot as plt
from fraud_detection_project.settings import BASE_DIR

def explore_data(df):
    """
    Conduct exploratory data analysis and generate visualizations.
    """
    # Basic statistics
    stats = df.describe()

    # Fraud count
    fraud_count = df['isFraud'].value_counts()

    # Define the directory for saving plots
    plot_dir = os.path.join(BASE_DIR, 'static')
    
    # Create the directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plotting the fraud counts
    plt.figure(figsize=(10, 5))
    fraud_count.plot(kind='bar')
    plt.title('Fraud vs Legitimate Transactions')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    
    # Save the plot
    plt.savefig(os.path.join(plot_dir, 'fraud_count_plot.png'))
    plt.close()

    return stats, fraud_count
