import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import json
import matplotlib.pyplot as plt
from ..lib.decorators import print_error

def status_analyze(data):
    try:
        num_clusters =5
        num_problematic_workstations = 3  # Adjust as needed

        show_craft=None #None, pq, pa
        relevant_set_nr = ['produced', 'off', 'short', 'long', 'working', 'availability', 'performance', 'quality', 'oee']
        relevant_set = relevant_set_nr +['workstation']
        # Load the production data
        # Display the first few rows and column descriptions
        # Preprocess the data
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Extract relevant features for clustering
        X = data[relevant_set]

        # Encode the 'workstation' column
        label_encoder = LabelEncoder()
        X['workstation_encoded'] = label_encoder.fit_transform(X['workstation'])

        # Perform clustering analysis
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # You can adjust the number of clusters as needed
        kmeans.fit(X.drop(columns=['workstation', 'workstation_encoded']))
        X['cluster'] = kmeans.labels_

        # Add workstation names to cluster results
        X['workstation'] = data['workstation']
        X['timestamp'] = data['timestamp']

        # Set minimum value to 0 for 'availability', 'performance', 'quality', and 'oee'
        X[relevant_set_nr] = X[relevant_set_nr].clip(lower=0)

        # Check for non-positive values in  key columns, all values need to positive for logarithm scale
        for key in ['availability', 'performance', 'quality', 'oee']:
            if (X[key] <= 0).any():
                X[key] = X[key].replace(0, 1e-6)  # Replace zeros with a small positive number
                X[key] = X[key].abs()  # Ensure all values are positive

        # Visualize the results
        if show_craft != None:

            plt.figure(figsize=(10, 6))

            # Scatter plot of 'performance' vs 'quality' with clusters color-coded
            for workstation_name in X['workstation'].unique():
                workstation_data = X[X['workstation'] == workstation_name]
                if show_craft =='pq':
                    plt.scatter(workstation_data['performance'], workstation_data['quality'],
                                label=f'Workstation {workstation_name}')
                elif show_craft=='pa':
                    plt.scatter(workstation_data['performance'], workstation_data['availability'],
                                label=f'Workstation {workstation_name}')
            plt.xlabel('Performance')
            if show_craft == 'pq':
                plt.ylabel('Quality')

            elif show_craft == 'pa':
                plt.ylabel('Availability')

            plt.title('Clustering Analysis')
            plt.legend()
            plt.grid(True)
            plt.show()
        # Display the cluster centers (average values of features) in tabular form
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_,
                                       columns=X.columns.drop(['workstation', 'cluster', 'workstation_encoded', 'timestamp']))
        representative_workstations = []
        representative_timestamps = []

        for i in range(kmeans.n_clusters):
            cluster_data = X[X['cluster'] == i]
            most_common_workstation = cluster_data['workstation'].mode()[0]
            median_timestamp = cluster_data['timestamp'].median()
            representative_workstations.append(most_common_workstation)
            representative_timestamps.append(median_timestamp)

        # Add the representative workstation and timestamp to the cluster centers
        cluster_centers['workstation'] = representative_workstations
        cluster_centers['timestamp'] = representative_timestamps


        # Identify the top 5 workstations whose efficiency should be increased
        bottom_5_workstations = cluster_centers.sort_values(by='oee', ascending=False).head(5)

        bottom_5_workstation_names = X.loc[X['workstation_encoded'].isin(bottom_5_workstations.index), 'workstation'].unique()
        bottom_5_workstations_arr =[]
        for v in bottom_5_workstation_names:
            bottom_5_workstations_arr.append({'workstation':v})
        # Identify workstation with potential issues
        problematic_workstations = cluster_centers.sort_values(by='oee').head(num_problematic_workstations)

        # Analyze which parameters are problematic for the identified workstation
        # Ensure we only check numeric columns
        return {
            "ClusterCenters": json.loads(cluster_centers.to_json(orient='records')),
            "Bottom5Workstations": bottom_5_workstations_arr,
            "ProblematicWorkstations": json.loads(problematic_workstations.to_json(orient='records'))
        }
    except Exception as e:
        print_error(e)